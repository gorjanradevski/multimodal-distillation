import io

import numpy as np
import torch
from PIL import Image
from yacs.config import CfgNode

from modelling.dataset_proto import ProtoDataset
from utils.data_utils import compile_transforms, get_video_transforms
from utils.samplers import get_sampler


class VideoDataset(ProtoDataset):
    def __init__(self, cfg: CfgNode, train: bool = False):
        self.cfg = cfg
        self.train = train
        self.dataset_name = (
            cfg.TRAIN_DATASET_NAME if self.train else cfg.VAL_DATASET_NAME
        )
        self.dataset_path = (
            cfg.TRAIN_DATASET_PATH if self.train else cfg.VAL_DATASET_PATH
        )
        self.dataset_type = self.cfg.DATASET_TYPE
        self.create_dataset()
        self.sampler = get_sampler[self.cfg.MODEL_NAME](cfg=cfg, train=train)

    def get_video_frames(self, **kwargs):
        indices = kwargs.pop("indices")
        video_id = kwargs.pop("video_id")
        # Fix for omnivore so we don't need to re-implement method
        if "resource" in kwargs:
            resource = kwargs.pop("resource")
        else:
            resource = self.resource
        # Epic-Kitchens
        if self.dataset_name == "EPIC-KITCHENS":
            narration_id = kwargs.pop("narration_id")
            unique_indices, inv_indices = np.unique(indices, return_inverse=True)
            frames = resource[video_id][narration_id][unique_indices]
            frames = [
                Image.open(io.BytesIO(frames[index]))
                for index in range(len(unique_indices))
            ]
            frames = [frames[index] for index in inv_indices]
        elif self.dataset_name == "something-something":
            unique_indices, inv_indices = np.unique(indices, return_inverse=True)
            frames = resource[video_id][unique_indices]
            frames = [
                Image.open(io.BytesIO(frames[index]))
                for index in range(len(unique_indices))
            ]
            frames = [frames[index] for index in inv_indices]
        else:
            raise ValueError(
                f"{self.dataset_type} cannot load anything with this dataset!"
            )
        return frames

    def get_flow_frames(self, **kwargs):
        indices = kwargs.pop("indices")
        video_id = kwargs.pop("video_id")
        # Fix for omnivore so we don't need to re-implement method
        if "resource" in kwargs:
            resource = kwargs.pop("resource")
        else:
            resource = self.resource

        if self.dataset_name == "EPIC-KITCHENS":
            narration_id = kwargs.pop("narration_id")

            indices_u = 2 * indices
            indices_v = indices_u + 1

            indices = np.empty(
                (indices_u.size + indices_v.size,), dtype=indices_u.dtype
            )
            indices[0::2] = indices_u
            indices[1::2] = indices_v

            unique_indices, inv_indices = np.unique(indices, return_inverse=True)

            frames = resource[video_id][narration_id][unique_indices]
            frames = [
                Image.open(io.BytesIO(frames[index]))
                for index in range(len(unique_indices))
            ]
            frames = [frames[index] for index in inv_indices]
            frames_u = frames[0::2]
            frames_v = frames[1::2]

            frames = [
                Image.merge(
                    "RGB", [frame_u, frame_v, Image.new("L", size=frame_v.size)]
                )
                for frame_u, frame_v in zip(frames_u, frames_v)
            ]
        elif self.dataset_name == "something-something":
            unique_indices, inv_indices = np.unique(indices, return_inverse=True)
            frames = resource[video_id][unique_indices]
            frames = [
                Image.open(io.BytesIO(frames[index]))
                for index in range(len(unique_indices))
            ]
            frames = [frames[index] for index in inv_indices]
        else:
            raise ValueError(
                f"{self.dataset_type} cannot load anything with this dataset!"
            )
        return frames

    def get_frames(self, **kwargs):
        if self.dataset_type == "video":
            return self.get_video_frames(**kwargs)
        elif self.dataset_type == "flow":
            return self.get_flow_frames(**kwargs)

    def __getitem__(self, idx: int):
        output = {self.dataset_type: []}
        if not hasattr(self, "resource"):
            self.open_resource()
        output["video_id"] = self.dataset[idx]["id"]
        if not hasattr(self, "indices"):
            indices = self.sampler(
                video_length=self.get_video_length(self.dataset[idx])
            )
        else:
            indices = self.indices
        # Check for existing transforms to achieve consistent teaching
        if not hasattr(self, "existing_transforms"):
            existing_transforms = {}
        else:
            existing_transforms = self.existing_transforms
        # Pass indices to the output object, so that the other datasets can use it
        output["indices"] = indices
        # If EPIC, we need the start frame, otherwise it is 0
        if self.dataset_name == "EPIC-KITCHENS":
            output["start_frame"] = self.dataset[idx]["start_frame"]
            output["narration_id"] = self.dataset[idx]["narration_id"]
        else:
            output["start_frame"] = 0
            output["narration_id"] = -1
        # Load all frames
        frames = self.get_frames(
            indices=indices,
            video_id=output["video_id"],
            narration_id=output["narration_id"],
        )
        # Aggregate frame
        output[self.dataset_type] = frames
        # Get the video transformation dictionary
        video_transforms_dict = get_video_transforms(
            augmentations_list=self.cfg.AUGMENTATIONS.get(self.dataset_type.upper()),
            train=self.train,
            cfg=self.cfg,
            existing_transforms=existing_transforms,
        )
        self.enforced_transforms = video_transforms_dict
        video_transforms = compile_transforms(video_transforms_dict)
        # Augment frames - (Eval_Clips x Frames) times
        for i in range(len(output[self.dataset_type])):
            # [Channels, Spatial size, Spatial size]
            frame = output[self.dataset_type][i]
            # [Channels, Spatial size, Spatial size]
            # or [Eval_Crops, Channels, Spatial size, Spatial size]
            output[self.dataset_type][i] = video_transforms(frame)
        # Stack the frames
        output[self.dataset_type] = torch.stack(output[self.dataset_type], dim=0)
        # [Eval_Clips x Frames, Channels, Spatial size, Spatial size]
        # or
        # [Eval_Clips x Frames, Eval_Crops, Channels, Spatial size, Spatial size]
        if len(output[self.dataset_type].shape) > 4:
            output[self.dataset_type] = output[self.dataset_type].permute(1, 0, 2, 3, 4)
        # [Eval_Crops, Eval_Clips x Frames, Channels, Spatial size, Spatial size]
        # Get spatial size
        spatial_size = output[self.dataset_type].size(-1)
        # Reshape
        output[self.dataset_type] = output[self.dataset_type].reshape(
            -1,  # Eval_Clips x Eval_Crops
            self.cfg.NUM_FRAMES,
            3,
            spatial_size,
            spatial_size,
        )
        # Obtain video labels
        output["labels"] = self.get_actions(self.dataset[idx])

        return output
