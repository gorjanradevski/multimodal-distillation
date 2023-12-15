import ast

import h5py
import numpy as np
import torch
from yacs.config import CfgNode

from modelling.dataset_proto import ProtoDataset
from utils.data_utils import fix_box
from utils.samplers import get_sampler


class LayoutDataset(ProtoDataset):
    def __init__(self, cfg: CfgNode, train: bool = False):
        self.cfg = cfg
        self.train = train
        self.dataset_name = (
            cfg.TRAIN_DATASET_NAME if self.train else cfg.VAL_DATASET_NAME
        )
        self.dataset_path = (
            cfg.TRAIN_DATASET_PATH if self.train else cfg.VAL_DATASET_PATH
        )
        self.create_dataset()
        self.cat2index = {"pad": 0, "object": 1, "hand": 2}
        self.max_objects = 14
        self.sampler = get_sampler[self.cfg.MODEL_NAME](cfg=cfg, train=train)

    def open_videos(self):
        raise AttributeError("Layout dataset does not use videos.")

    def open_resource(self):
        assert self.cfg.DETECTIONS_PATH, "Path to detections must be provided!"
        self.resource = h5py.File(
            self.cfg.DETECTIONS_PATH, "r", libver="latest", swmr=True
        )

    def get_detections(self, dataset_element, index: int, **kwargs):
        boxes, class_labels, scores, states, sides = [], [], [], [], []
        if self.dataset_name == "EPIC-KITCHENS":
            if "resource" in kwargs:
                resource = kwargs.pop("resource")
            else:
                resource = self.resource
            video_name = dataset_element["id"]
            detections = ast.literal_eval(
                str(np.array(resource[video_name][str(index)]))[2:-1]
            )
            for e in detections:
                boxes.append(e["box"])
                class_labels.append(self.cat2index[e["category"]])
                scores.append(e["score"])
                state = e["state"] + 1 if "state" in e else 0
                states.append(state)
                side = e["side"] + 1 if "side" in e else 0
                sides.append(side)
        elif self.dataset_name == "something-something":
            w, h = dataset_element["size"]
            frame_objects = dataset_element["frames"][index]
            for e in frame_objects:
                box = fix_box([e["x1"], e["y1"], e["x2"], e["y2"]], video_size=(h, w))
                boxes.append(box)
                class_label = (
                    self.cat2index["hand"]
                    if "hand" in e["category"]
                    else self.cat2index["object"]
                )
                class_labels.append(class_label)
                scores.append(e["score"])
            sides = [0 for _ in range(len(class_labels))]
            states = [0 for _ in range(len(class_labels))]
        else:
            raise ValueError(f"{self.dataset_name} not available!")

        return boxes, class_labels, scores, sides, states

    def get_video_length(self, sample):  # Reimplemented
        return len(sample["frames"])

    def __getitem__(self, idx: int):
        if not hasattr(self, "resource") and self.dataset_name == "EPIC-KITCHENS":
            self.open_resource()
        output = {
            "bboxes": [],
            "class_labels": [],
            "scores": [],
            "sides": [],
            "states": [],
            "src_key_padding_mask_boxes": [],
        }
        if not hasattr(self, "indices"):
            indices = self.sampler(
                video_length=self.get_video_length(self.dataset[idx])
            )
        else:
            indices = self.indices
        output["indices"] = indices
        output["start_frame"] = 0

        for index in indices:
            bboxes, class_labels, scores, sides, states = self.get_detections(
                self.dataset[idx], index
            )
            # Perform padding to max objects
            while len(bboxes) < self.max_objects:
                class_labels.append(self.cat2index["pad"])
                bboxes.append([0, 0, 1e-9, 1e-9])
                scores.append(0.0)
                sides.append(0)
                states.append(0)
            # Add boxes
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            if self.dataset_name == "something-something":
                w, h = self.dataset[idx]["size"]
                bboxes = bboxes / torch.tensor([w, h, w, h])
            output["bboxes"].append(bboxes)
            # Add class labels
            output["class_labels"].append(torch.tensor(class_labels))
            # Add scores
            output["scores"].append(torch.tensor(scores))
            # Add sides
            output["sides"].append(torch.tensor(sides))
            # Add states
            output["states"].append(torch.tensor(states))
            # Generate mask
            output["src_key_padding_mask_boxes"].append(
                output["class_labels"][-1] == self.cat2index["pad"]
            )
        # Convert to tensors
        output = {
            key: torch.stack(val, dim=0)
            if key not in ["indices", "start_frame"]
            else val
            for key, val in output.items()
        }
        output["labels"] = self.get_actions(self.dataset[idx])

        return output
