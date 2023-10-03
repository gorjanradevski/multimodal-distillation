import argparse
import logging
from collections import OrderedDict
from os.path import join as pjoin

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from yacs.config import CfgNode

from modelling.datasets import dataset_factory
from modelling.models import model_factory
from utils.calibration import CalibrationEvaluator
from utils.data_utils import separator
from utils.evaluation import evaluators_factory
from utils.setup import get_cfg_defaults


def unwrap_compiled_checkpoint(checkpoint):
    # If Pytorch 2.0, checkpoint is wrapped with _orig_mod, so we remove it
    new_checkpoint = OrderedDict()
    for key in checkpoint.keys():
        new_key = key[10:] if key.startswith("_orig_mod") else key
        new_checkpoint[new_key] = checkpoint[key]

    return new_checkpoint


@torch.no_grad()
def inference(cfg: CfgNode):
    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator()
    # Prepare datasets
    if accelerator.is_main_process:
        logging.info("Preparing datasets...")
    # Prepare validation dataset
    if accelerator.is_main_process:
        logging.info(separator)
        logging.info(f"The config is:\n{cfg}")
        logging.info(separator)
    val_dataset = dataset_factory[cfg.DATASET_TYPE](cfg, train=False)
    if accelerator.is_main_process:
        logging.info(f"Validating on {len(val_dataset)}")
    # Prepare loaders
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
    checkpoint = torch.load(
        pjoin(cfg.EXPERIMENT_PATH, "model_checkpoint.pt"), map_location="cpu"
    )
    checkpoint = unwrap_compiled_checkpoint(checkpoint)
    model.load_state_dict(checkpoint)
    # Prepare evaluators
    evaluator = evaluators_factory[cfg.VAL_DATASET_NAME](len(val_dataset), cfg)
    calibration_evaluator = CalibrationEvaluator(cfg)
    if accelerator.is_main_process:
        logging.info("Starting inference...")
    # Accelerate
    model, val_loader = accelerator.prepare(model, val_loader)
    model.train(False)
    evaluator.reset()
    # Inference
    for batch in tqdm(val_loader, disable=not accelerator.is_main_process):
        # Obtain outputs: [b * n_clips, n_actions]
        model_output = model(batch)
        # Gather
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
        # Evaluate
        evaluator.process(all_outputs, all_labels)
        calibration_evaluator.process(all_outputs, all_labels)
    # Metrics
    if accelerator.is_main_process:
        metrics = evaluator.evaluate_verbose()
        for m in metrics.keys():
            logging.info(f"{m}: {metrics[m]}")
        # Calibration Metrics
        calibration_metrics = calibration_evaluator.evaluate()
        logging.info("=============== Calibration Metrics (ECE) ===============")
        for m in calibration_metrics.keys():
            logging.info(f"{m}: {calibration_metrics[m]}")


def main():
    parser = argparse.ArgumentParser(description="Inference.")
    parser.add_argument(
        "--experiment_path", required=True, help="Path to the experiment."
    )
    parser.add_argument("--opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(pjoin(args.experiment_path, "config.yaml"))
    if args.opts:
        cfg.merge_from_list(args.opts)
    # Set backbone_model_path to None as not important
    cfg.BACKBONE_MODEL_PATH = None
    # Freeze the config
    cfg.freeze()
    inference(cfg)


if __name__ == "__main__":
    main()
