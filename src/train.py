import logging
import random
from os.path import join as pjoin

import torch
from accelerate import Accelerator
from torch import optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from yacs.config import CfgNode

from modelling.datasets import dataset_factory
from modelling.losses import LossesModule
from modelling.models import model_factory
from utils.data_utils import separator
from utils.evaluation import evaluators_factory
from utils.setup import train_setup
from utils.train_utils import get_linear_schedule_with_warmup


def train(cfg: CfgNode, accelerator: Accelerator):
    if cfg.LOG_TO_FILE:
        if accelerator.is_main_process:
            logging.basicConfig(
                level=logging.INFO,
                filename=pjoin(cfg.EXPERIMENT_PATH, "experiment_log.log"),
                filemode="w",
            )
    else:
        logging.basicConfig(level=logging.INFO)
    if accelerator.is_main_process:
        logging.info(separator)
        logging.info(f"The config file is:\n {cfg}")
        logging.info(separator)
    # Prepare datasets
    if accelerator.is_main_process:
        logging.info("Preparing datasets...")
    # Prepare train dataset
    train_dataset = dataset_factory[cfg.DATASET_TYPE](cfg, train=True)
    # Prepare validation dataset
    val_dataset = dataset_factory[cfg.DATASET_TYPE](cfg, train=False)
    num_training_samples = len(train_dataset)
    if cfg.VAL_SUBSET:
        val_indices = random.sample(range(len(val_dataset)), cfg.VAL_SUBSET)
        val_dataset = Subset(val_dataset, val_indices)
    num_validation_samples = len(val_dataset)
    if accelerator.is_main_process:
        logging.info(f"Training on {num_training_samples}")
        logging.info(f"Validating on {num_validation_samples}")
    # Prepare loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True if cfg.NUM_WORKERS else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True if cfg.NUM_WORKERS else False,
    )
    if accelerator.is_main_process:
        logging.info("Preparing model...")
    # Prepare model
    model = model_factory[cfg.MODEL_NAME](cfg)
    # If PyTorch 2.0, compile the model
    if hasattr(torch, "compile"):
        if accelerator.is_main_process:
            logging.info("Compile model...")
        model = torch.compile(model)
    # Optimizer, scheduler and similar...
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    num_batches = num_training_samples // cfg.BATCH_SIZE
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.WARMUP_EPOCHS * num_batches,
        num_training_steps=cfg.EPOCHS * num_batches,
    )
    evaluator = evaluators_factory[cfg.VAL_DATASET_NAME](num_validation_samples, cfg)
    # Accelerate
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    # Create loss
    criterion = LossesModule(cfg)
    if accelerator.is_main_process:
        logging.info("Starting training...")
    for epoch in range(cfg.EPOCHS):
        # Training loop
        model.train(True)
        with tqdm(
            total=len(train_loader), disable=not accelerator.is_main_process
        ) as pbar:
            for batch in train_loader:
                # Remove past gradients
                optimizer.zero_grad()
                # Obtain outputs: [b * n_clips, n_actions]
                model_output = model(batch)
                # Measure loss and update weights
                loss = criterion(model_output, batch)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), cfg.CLIP_VAL)
                optimizer.step()
                # Update the scheduler
                scheduler.step()
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"Loss": loss.item()})
        # Validation loop
        model.train(False)
        evaluator.reset()
        for batch in tqdm(val_loader, disable=not accelerator.is_main_process):
            with torch.no_grad():
                # Obtain outputs: [b * n_clips, n_actions]
                model_output = model(batch)
                all_outputs = accelerator.gather(model_output)
                all_labels = accelerator.gather(batch["labels"])
                # Reshape outputs and put on cpu
                for key in all_outputs.keys():
                    num_classes = all_outputs[key].size(-1)
                    # Reshape
                    all_outputs[key] = all_outputs[key].reshape(
                        -1, cfg.NUM_TEST_CLIPS * cfg.NUM_TEST_CROPS, num_classes
                    )
                    # Move on CPU
                    all_outputs[key] = all_outputs[key].cpu()
                # Put labels on cpu
                for key in all_labels.keys():
                    all_labels[key] = all_labels[key].cpu()
            # Pass to evaluator
            evaluator.process(all_outputs, all_labels)
        # Evaluate & save model
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            metrics = evaluator.evaluate()
            if evaluator.is_best():
                logging.info(separator)
                logging.info(f"Found new best on epoch {epoch+1}!")
                logging.info(separator)
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(
                    unwrapped_model.state_dict(),
                    pjoin(cfg.EXPERIMENT_PATH, "model_checkpoint.pt"),
                )
            for m in metrics.keys():
                logging.info(f"{m}: {metrics[m]}")


def main():
    cfg, accelerator = train_setup("Trains an action recognition model.")
    train(cfg, accelerator)


if __name__ == "__main__":
    main()
