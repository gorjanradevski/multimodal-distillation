import logging
import random
from os.path import join as pjoin

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from yacs.config import CfgNode

from modelling.datasets import dataset_factory
from modelling.distiller import DistillationCriterion, TeacherEnsemble
from modelling.models import model_factory
from utils.data_utils import separator
from utils.evaluation import evaluators_factory
from utils.setup import train_setup
from utils.train_utils import (
    EpochHandler,
    get_linear_schedule_with_warmup,
    move_batch_to_device,
)


@torch.no_grad()
def set_teacher_weights(
    cfg: CfgNode, teacher: nn.Module, loader: DataLoader, accelerator: Accelerator
):
    teacher = teacher.to(cfg.DEVICE)
    total = 0
    for batch in tqdm(loader, disable=not accelerator.is_main_process):
        batch = move_batch_to_device(batch, device=cfg.DEVICE)
        teacher_logits = teacher.get_teacher_logits(batch)
        for action_name in teacher_logits.keys():
            # [Batch_Size, Num_Teachers, Num_Classes]
            logits = torch.cat(
                [t_o.unsqueeze(1) for t_o in teacher_logits[action_name]], dim=1
            )
            losses = teacher.get_losses_wrt_labels(logits, batch["labels"][action_name])
            teacher.weights[action_name] = teacher.weights[action_name] + losses.sum(0)
        # Measure number of batches
        total += losses.size(0)
    # Find average weights
    for action_name in teacher.weights.keys():
        teacher.weights[action_name] = teacher.weights[action_name] / total
        # Convert to weights - normalize
        teacher.weights[action_name] = F.softmin(
            teacher.weights[action_name] / cfg.WEIGHTS_TEMPERATURE, dim=-1
        )
    # Return to CPU
    teacher = teacher.cpu()

    return teacher


def distill(cfg: CfgNode, accelerator: Accelerator):
    if cfg.LOG_TO_FILE:
        if accelerator.is_main_process:
            logging.basicConfig(
                level=logging.INFO,
                filename=pjoin(cfg.EXPERIMENT_PATH, "experiment_log.log"),
                filemode="a",
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
    num_training_samples = len(train_dataset)
    # Prepare validation dataset
    val_dataset = dataset_factory[cfg.DATASET_TYPE](cfg, train=False)
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
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True if cfg.NUM_WORKERS else False,
    )
    if accelerator.is_main_process:
        logging.info("Preparing teacher...")
    teacher = TeacherEnsemble(cfg)
    # Check if weighted distillation
    if cfg.DISTILLATION_WEIGHTING_SCHEME == "full-dataset":
        if accelerator.is_main_process:
            logging.info("Weighted distillation, setting weights...")
        import json

        train_indices_path = pjoin(
            cfg.TEACHERS.RGB_TEACHER_EXPERIMENT_PATH, "train_indices.json"
        )
        train_indices = set(json.load(open(train_indices_path, "r")))
        reverse_indices = [
            i for i in range(len(train_dataset)) if i not in train_indices
        ]
        weighting_dataset = Subset(train_dataset.set_weighted(), reverse_indices)
        weighting_loader = DataLoader(
            weighting_dataset,
            shuffle=False,
            batch_size=cfg.BATCH_SIZE,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True if cfg.NUM_WORKERS else False,
        )
        teacher = set_teacher_weights(
            cfg=cfg, teacher=teacher, loader=weighting_loader, accelerator=accelerator
        )
        if accelerator.is_main_process:
            for action_name in teacher.weights.keys():
                weights = teacher.weights[action_name]
                logging.info(f"For action {action_name}, the weights are: {weights}")
    # Wait for all devices to calibrate...
    accelerator.wait_for_everyone()
    # Preparing student
    if accelerator.is_main_process:
        logging.info("Preparing student...")
    # Prepare model
    student = model_factory[cfg.MODEL_NAME](cfg)
    criterion = DistillationCriterion(cfg)
    # Optimizer, scheduler, evaluator & loss
    optimizer = optim.AdamW(
        student.parameters(),
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
    (
        student,
        teacher,
        optimizer,
        train_loader,
        val_loader,
        scheduler,
    ) = accelerator.prepare(
        student, teacher, optimizer, train_loader, val_loader, scheduler
    )
    # https://huggingface.co/docs/accelerate/quicktour#savingloading-entire-states
    # Check for starting from existing checkpoint
    epoch_handler = EpochHandler()
    if cfg.WARM_RESTART:
        if accelerator.is_main_process:
            logging.info("Performing Warm Restart!")
        accelerator.load_state(pjoin(cfg.EXPERIMENT_PATH, "full-checkpoint"))
        epoch_handler.load_state(pjoin(cfg.EXPERIMENT_PATH, "last-epoch"))
    # Starting training
    if accelerator.is_main_process:
        logging.info("Starting training...")
    for epoch in range(epoch_handler.epoch, cfg.EPOCHS):
        # Training loop
        student.train(True)
        with tqdm(
            total=len(train_loader), disable=not accelerator.is_main_process
        ) as pbar:
            for batch in train_loader:
                # Remove past gradients
                optimizer.zero_grad()
                # Get outputs
                student_logits = student(batch)
                teacher_logits = teacher(batch)
                # Measure loss
                loss = criterion(student_logits, teacher_logits, batch)
                # Backpropagate
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(student.parameters(), cfg.CLIP_VAL)
                optimizer.step()
                # Update the scheduler
                scheduler.step()
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"Loss": loss.item()})
        # Validation loop
        student.train(False)
        evaluator.reset()
        for batch in tqdm(val_loader, disable=not accelerator.is_main_process):
            with torch.no_grad():
                # Obtain outputs: [b * n_clips, n_actions]
                student_output = student(batch)
                all_outputs = accelerator.gather(student_output)
                all_labels = accelerator.gather(batch["labels"])
                # Reshape outputs & put on cpu
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
                unwrapped_student = accelerator.unwrap_model(student)
                accelerator.save(
                    unwrapped_student.state_dict(),
                    pjoin(cfg.EXPERIMENT_PATH, "model_checkpoint.pt"),
                )
            for m in metrics.keys():
                logging.info(f"{m}: {metrics[m]}")
        # Update epoch handler
        epoch_handler.set_epoch(epoch + 1)
        # Save/Overwrite full checkpoint
        logging.info(f"Saving/Overwriting full checkpoint at epoch {epoch+1}")
        accelerator.save_state(pjoin(cfg.EXPERIMENT_PATH, "full-checkpoint"))
        epoch_handler.save_state(pjoin(cfg.EXPERIMENT_PATH, "last-epoch"))


def main():
    cfg, accelerator = train_setup("Performs multimodal knowledge distillation.")
    distill(cfg, accelerator)


if __name__ == "__main__":
    main()
