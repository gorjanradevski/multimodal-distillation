import argparse
import os

from accelerate import Accelerator, DistributedDataParallelKwargs
from yacs.config import CfgNode as CN

_C = CN()
# Warm restarting training
_C.WARM_RESTART = False
# Whether to log to a file
_C.LOG_TO_FILE = False
# The path to the config
_C.CONFIG_PATH = None
# The name of the model to be used
_C.MODEL_NAME = None
# IF Resnet3D, then model depth too
_C.RESNET3D_DEPTH = None
# Path to the train dataset - usually a json file
_C.TRAIN_DATASET_PATH = None
# Path to the val dataset - usually a json file
_C.VAL_DATASET_PATH = None
# The name of the training dataset
_C.TRAIN_DATASET_NAME = None
# The name of the validation dataset
_C.VAL_DATASET_NAME = None
# Path to the labels
_C.LABELS_PATH = None
# Version of the dataset: Relevant for Epic Kitchens
_C.DATASET_VERSION = None
# Type of the dataset: Can be video or layout
_C.DATASET_TYPE = None
# Train subset
_C.TRAIN_SUBSET = None
# Validation subset
_C.VAL_SUBSET = None
# Whether the data during training is omnivore and model
_C.OMNIVORE = None
# Path to the videos
_C.VIDEOS_PATH = None
# Meant to modify the participants list (unseen list)
_C.EPIC_PARTICIPANTS = None
# Path to the flow
_C.FLOW_PATH = None
# Path to the depth
_C.DEPTH_PATH = None
# Path to the skeleton
_C.SKELETON_PATH = None
# Path to the segmentations
_C.SEGMENTATION_PATH = None
# Path to frame mapping dictionary (used for the Visor dataset in Epic-Kitchens)
_C.SEGMENTATION_FRAME_MAPPING_PATH = None
# Path to the detections (only used when Epic-Kitchens)
_C.DETECTIONS_PATH = None
# Number of frames to be sampled from the video
_C.NUM_FRAMES = 8
# Number of frames the student will use
_C.STUDENT_SUBSAMPLE_RATE = None
# Whether to use uniform sampling of frames
_C.FRAME_UNIFORM = False
# The batch size
_C.BATCH_SIZE = 12
# The learning rate
_C.LEARNING_RATE = 0.0001
# The weight decay
_C.WEIGHT_DECAY = 0.0001
# The gradient clipping value
_C.CLIP_VAL = 5.0
# The number of workers - threads in the dataloader
_C.NUM_WORKERS = 0
# The number of epochs for training
_C.EPOCHS = 20
# The number of warmup epochs for the warmup optimizer
_C.WARMUP_EPOCHS = 2
# Path to a pre-trained backbone
_C.BACKBONE_MODEL_PATH = None
# Path to existing checkpoint
_C.CHECKPOINT_PATH = None
# Device used for training: Cuda or Cpu
_C.DEVICE = "cuda"
# Where to save the experiment
_C.EXPERIMENT_PATH = "experiments/default"
# Whether to freeze the backbone during training
_C.FREEZE_BACKBONE = False
# Number of testing clips when evaluating
_C.NUM_TEST_CLIPS = 1
# How many crops during inference: Either 1 or 3
_C.NUM_TEST_CROPS = 1
# Stride for getting frames
_C.STRIDE = 8
# The augmentations per dataset
_C.AUGMENTATIONS = CN()
_C.AUGMENTATIONS.VIDEO = ["IdentityTransform"]
_C.AUGMENTATIONS.FLOW = ["IdentityTransform"]
_C.AUGMENTATIONS.DEPTH = ["IdentityTransform"]
_C.AUGMENTATIONS.SKELETON = ["IdentityTransform"]
_C.AUGMENTATIONS.LAYOUT = ["IdentityTransform"]
_C.AUGMENTATIONS.AUDIO = ["IdentityTransform"]
_C.AUGMENTATIONS.SEGMENTATION = ["IdentityTransform"]
# The number of actions, for each action type
_C.TOTAL_ACTIONS = CN()
_C.TOTAL_ACTIONS.ACTION = None
_C.TOTAL_ACTIONS.NOUN = None
_C.TOTAL_ACTIONS.VERB = None
# The action weights, for each action type
_C.ACTION_WEIGHTS = CN()
_C.ACTION_WEIGHTS.ACTION = None
_C.ACTION_WEIGHTS.NOUN = None
_C.ACTION_WEIGHTS.VERB = None
# The losses weights, ONLY when distillation
_C.LOSS_WEIGHTS = CN()
_C.LOSS_WEIGHTS.DISTILLATION = 1.0
_C.LOSS_WEIGHTS.GROUND_TRUTH = 0.0
# Path to the teacher experiment path
_C.TEACHERS = CN()
_C.TEACHERS.OBJ_TEACHER_EXPERIMENT_PATH = None
_C.TEACHERS.RGB_TEACHER_EXPERIMENT_PATH = None
_C.TEACHERS.FLOW_TEACHER_EXPERIMENT_PATH = None
_C.TEACHERS.DEPTH_TEACHER_EXPERIMENT_PATH = None
_C.TEACHERS.SKELETON_TEACHER_EXPERIMENT_PATH = None
_C.TEACHERS.AUDIO_TEACHER_EXPERIMENT_PATH = None
_C.TEACHERS.SEGMENTATION_TEACHER_EXPERIMENT_PATH = None
_C.CALIBRATE_TEACHER = False
# Can be None, per-sample, full-dataset
_C.DISTILLATION_WEIGHTING_SCHEME = None
# The temperature (used during distilation)
_C.TEMPERATURE = 1.0
# The weighting temperature (used during distillation)
_C.WEIGHTS_TEMPERATURE = 1.0
# Audio stuff
_C.AUDIO_PATH = None
# Video FPS
_C.VIDEO_FPS = 60
# Audio sample rate
_C.AUDIO_SAMPLE_RATE = 24000
# Length of audio segments extracted from action sequence (in seconds)
_C.AUDIO_SEGMENT_LENGTH = 1.116  # to get 224 width of spectrogram
# Audio resample rate
_C.AUDIO_RESAMPLE_RATE = None
# Evaluate for tail class indices (on EPIC)
_C.EPIC_TAIL_NOUNS_PATH = None
_C.EPIC_TAIL_VERBS_PATH = None
_C.TEST_SET_INFERENCE = False


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


def train_setup(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to the config file."
    )
    parser.add_argument("--opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_path)
    cfg.CONFIG_PATH = args.config_path
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    # Prepare accelerator
    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        cpu=cfg.DEVICE == "cpu",
    )
    # Saving logic
    if accelerator.is_main_process:
        # If we have experiment already & not restarting --> Error!
        if os.path.exists(cfg.EXPERIMENT_PATH) and not cfg.WARM_RESTART:
            raise ValueError(
                f"{cfg.EXPERIMENT_PATH} exists & WARM_RESTART is False!\n"
                f"Please delete {cfg.EXPERIMENT_PATH} and run again!"
            )
        # If we are restarting, we have to have experiment!
        if cfg.WARM_RESTART:
            assert os.path.exists(
                cfg.EXPERIMENT_PATH
            ), f"There is no {cfg.EXPERIMENT_PATH} to restart from!"
        else:
            # Otherwise, we create the experiment directory
            os.makedirs(cfg.EXPERIMENT_PATH, exist_ok=False)
        with open(os.path.join(cfg.EXPERIMENT_PATH, "config.yaml"), "w") as f:
            f.write(cfg.dump())

    return cfg, accelerator
