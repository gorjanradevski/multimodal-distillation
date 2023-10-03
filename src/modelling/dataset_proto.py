import csv
import json
import pickle
import re
from typing import Dict

import h5py
import torch
from torch.utils.data import Dataset
from yacs.config import CfgNode

from utils.samplers import get_sampler


class ProtoDataset(Dataset):
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

    def create_dataset(self):
        self.dataset = []
        if self.dataset_name == "something-something":
            assert (
                self.cfg.LABELS_PATH
            ), "If something-something, path to labels required"
            self.dataset = json.load(open(self.dataset_path))
            self.labels = json.load(open(self.cfg.LABELS_PATH))
        elif self.dataset_name == "charades" or self.dataset_name == "charades-ego":
            with open(self.dataset_path, newline="") as csvfile:
                for row in csv.DictReader(csvfile):
                    if len(row["actions"]) == 0:
                        continue
                    if row["id"] in {"93MIK", "8NJQ3", "8U2AR"}:
                        continue
                    actions = [int(a[1:4]) for a in row["actions"].split(";")]
                    self.dataset.append({"id": row["id"], "actions": actions})
        elif self.dataset_name == "egtea-gaze":
            with open(self.dataset_path, "r") as text_file:
                for line in text_file.readlines():
                    video_id, action_id = line.split()[:2]
                    # Action start from 1, subtracting 1
                    self.dataset.append({"id": video_id, "actions": int(action_id) - 1})
        elif self.dataset_name == "montalbano":
            self.dataset = json.load(open(self.dataset_path))
        elif self.dataset_name == "EPIC-KITCHENS":
            # Flow: https://github.com/epic-kitchens/epic-kitchens-download-scripts/issues/17#issuecomment-1222288006
            assert (
                self.cfg.DATASET_VERSION == 55 or self.cfg.DATASET_VERSION == 100
            ), "If EPIC-KITCHENS, dataset version must be provided (55 or 100)"
            if self.cfg.DATASET_VERSION == 55:
                data_file = pickle.load(open(self.dataset_path, "rb"))
                # FIXME: Removing two indices which are bad
                bad = {36787, 36788}
                for index in data_file.index.to_list():
                    if index in bad:
                        continue
                    self.dataset.append(
                        {
                            "id": data_file["video_id"][index],
                            "narration_id": data_file["narration_id"][index],
                            "start_frame": data_file["start_frame"][index],
                            "stop_frame": data_file["stop_frame"][index],
                            "start_timestamp": data_file["start_timestamp"][index],
                            "stop_timestamp": data_file["stop_timestamp"][index],
                            # Noun starts from 1, subtracting 1
                            "noun_class": data_file["noun_class"][index] - 1,
                            "verb_class": data_file["verb_class"][index],
                        }
                    )
            elif self.cfg.DATASET_VERSION == 100:
                with open(self.dataset_path, newline="") as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if self.cfg.EPIC_PARTICIPANTS:
                            if row["participant_id"] not in self.cfg.EPIC_PARTICIPANTS:
                                continue
                        self.dataset.append(
                            {
                                "id": row["video_id"],
                                "narration_id": row["narration_id"],
                                "start_frame": int(row["start_frame"]),
                                "stop_frame": int(row["stop_frame"]),
                                "start_timestamp": row["start_timestamp"],
                                "stop_timestamp": row["stop_timestamp"],
                                "noun_class": int(row["noun_class"]),
                                "verb_class": int(row["verb_class"]),
                            }
                        )
        else:
            raise ValueError(f"{self.dataset_name} does not exist!")

    def get_actions(self, sample) -> Dict[str, torch.Tensor]:
        if self.dataset_name == "something-something":
            actions = {
                "ACTION": torch.tensor(
                    int(self.labels[re.sub("[\[\]]", "", sample["template"])])
                )
            }
        elif self.dataset_name == "charades" or self.dataset_name == "charades-ego":
            actions = torch.zeros(self.cfg.TOTAL_ACTIONS.ACTION, dtype=torch.float)
            actions[sample["ACTION"]] = 1.0
            actions = {"ACTION": actions}
        elif self.dataset_name == "egtea-gaze":
            actions = {"ACTION": torch.tensor(sample["actions"])}
        elif self.dataset_name == "montalbano":
            actions = {"ACTION": torch.tensor(sample["gesture_class"] - 1)}
        elif self.dataset_name == "EPIC-KITCHENS":
            actions = {
                "NOUN": torch.tensor(sample["noun_class"]),
                "VERB": torch.tensor(sample["verb_class"]),
            }
        else:
            raise ValueError(f"{self.dataset_name} does not exist!")

        return actions

    def __len__(self):
        return len(self.dataset)

    def open_resource(self):
        if self.dataset_type == "video":
            self.resource = h5py.File(
                self.cfg.VIDEOS_PATH, "r", libver="latest", swmr=True
            )
        elif self.dataset_type == "flow":
            self.resource = h5py.File(
                self.cfg.FLOW_PATH, "r", libver="latest", swmr=True
            )
        elif self.dataset_type == "audio":
            self.resource = h5py.File(
                self.cfg.AUDIO_PATH, "r", libver="latest", swmr=True
            )
        elif self.dataset_type == "segmentation":
            self.resource = h5py.File(
                self.cfg.SEGMENTATION_PATH, "r", libver="latest", swmr=True
            )
        else:
            raise ValueError(
                f"{self.dataset_type} cannot load anything with this dataset!"
            )

    def get_video_length(self, sample):
        # EPIC (covers audio too - hack)
        if self.dataset_name == "EPIC-KITCHENS" or self.dataset_name == "montalbano":
            return sample["stop_frame"] - sample["start_frame"] + 1
        # All else
        return len(self.resource[sample["id"]])

    def set_indices(self, indices):
        self.indices = indices

    def set_existing_transforms(self, transforms):
        self.existing_transforms = transforms

    def __getitem__(self, idx: int):
        raise NotImplementedError("Subclasses must implement '__getitem__'.")
