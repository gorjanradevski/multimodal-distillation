from collections import OrderedDict
from typing import Dict

import torch
from torch import nn
from torchvision.transforms import functional as TF
from yacs.config import CfgNode

from modelling.hand_models import Stlt
from modelling.resnets3d import generate_model
from modelling.swin import SwinTransformer3D
from utils.data_utils import get_normalizer


class R3d(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(R3d, self).__init__()
        self.cfg = cfg
        resnet = generate_model(model_depth=self.cfg.RESNET3D_DEPTH, n_classes=700)
        if self.cfg.BACKBONE_MODEL_PATH:
            checkpoint = torch.load(cfg.BACKBONE_MODEL_PATH, map_location="cpu")
            resnet.load_state_dict(checkpoint["state_dict"])
            # Freeze the BatchNorm3D layers
            for module in resnet.modules():
                if isinstance(module, nn.BatchNorm3d):
                    module.weight.requires_grad = False
                    module.bias.requires_grad = False
        # Strip the last two layers (pooling & classifier)
        self.resnet = torch.nn.Sequential(*(list(resnet.children())[:-2]))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        if self.cfg.RESNET3D_DEPTH == 18:
            self.projector = nn.Sequential(nn.Linear(512, 2048), nn.ReLU())
        # Build classifier
        self.classifiers = nn.ModuleDict(
            {
                actions_name: nn.Linear(2048, actions_num)
                for actions_name, actions_num in self.cfg.TOTAL_ACTIONS.items()
                if actions_num is not None
            }
        )
        # Load existing checkpoint, if any
        if cfg.CHECKPOINT_PATH:
            self.load_state_dict(torch.load(cfg.CHECKPOINT_PATH, map_location="cpu"))

    def train(self, mode: bool):
        super(R3d, self).train(mode)
        if self.cfg.BACKBONE_MODEL_PATH:
            for module in self.resnet.modules():
                if isinstance(module, nn.BatchNorm3d):
                    module.train(False)

    def get_modality(self):
        mapping = {
            "video": "video",
            "video_flow": "video",
            "flow": "flow",
            "depth": "depth",
            "video_layout": "video",
            "video_depth": "video",
            "layout": "layout",
            "video_layout_flow": "video",
            "omnivore": "omnivore",
            "audio": "audio",
            "video_audio": "video",
            "video_flow_audio": "video",
            "segmentation": "segmentation",
            "video_segmentation": "video",
        }

        return mapping[self.cfg.DATASET_TYPE]

    def forward(self, batch: Dict[str, torch.Tensor]):
        # Obtain the modality
        modality = self.get_modality()
        # Get the video frames and prepare
        video_frames = batch[modality]
        # Normalize video frames
        normalizer = get_normalizer(input_type=modality, model_name="resnet3d")
        video_frames = normalizer(video_frames)
        b, n_clips, n_frames, c, s, s = video_frames.size()
        # print(video_frames.size())
        video_frames = video_frames.reshape(b * n_clips, n_frames, c, s, s)
        # HACK: Resize the video frames to 112 in case they're not already
        if s > 112:
            video_frames = video_frames.view(-1, c, s, s)
            video_frames = TF.resize(video_frames, size=(112, 112), antialias=True)
            video_frames = video_frames.view(b * n_clips, n_frames, c, 112, 112)
        video_frames = video_frames.permute(0, 2, 1, 3, 4)
        # Extract features
        output = {}
        features = self.avgpool(self.resnet(video_frames)).flatten(1)
        features = features.contiguous()
        if self.cfg.RESNET3D_DEPTH == 18:
            features = self.projector(features)
        # Classify
        for actions_name in self.classifiers.keys():
            output[actions_name] = self.classifiers[actions_name](features)

        return output


class SwinModel(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(SwinModel, self).__init__()
        self.cfg = cfg
        # Create backbone
        self.backbone = SwinTransformer3D(
            patch_size=(2, 4, 4),
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=(8, 7, 7),
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            patch_norm=True,
        )
        if self.cfg.BACKBONE_MODEL_PATH:
            checkpoint = torch.load(self.cfg.BACKBONE_MODEL_PATH, map_location="cpu")
            new_state_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                if "backbone" in k:
                    name = k[9:]
                    new_state_dict[name] = v
            self.backbone.load_state_dict(new_state_dict)
        # Build classifier
        self.classifiers = nn.ModuleDict(
            {
                actions_name: nn.Linear(768, actions_num)
                for actions_name, actions_num in self.cfg.TOTAL_ACTIONS.items()
                if actions_num is not None
            }
        )
        # Load existing checkpoint, if any
        if cfg.CHECKPOINT_PATH:
            self.load_state_dict(torch.load(cfg.CHECKPOINT_PATH, map_location="cpu"))

    def get_modality(self):
        mapping = {
            "video": "video",
            "video_flow": "video",
            "flow": "flow",
            "depth": "depth",
            "video_layout": "video",
            "video_depth": "video",
            "layout": "layout",
            "video_layout_flow": "video",
            "omnivore": "omnivore",
            "audio": "audio",
            "video_audio": "video",
            "video_flow_audio": "video",
            "segmentation": "segmentation",
            "video_segmentation": "video",
        }

        return mapping[self.cfg.DATASET_TYPE]

    def forward(self, batch: Dict[str, torch.Tensor]):
        # Obtain the modality
        modality = self.get_modality()
        # Get the video frames and prepare
        video_frames = batch[modality]
        # Normalize video frames
        normalizer = get_normalizer(input_type=modality, model_name="swin")
        video_frames = normalizer(video_frames)
        b, n_clips, n_frames, c, s, s = video_frames.size()
        video_frames = video_frames.reshape(b * n_clips, n_frames, c, s, s)
        video_frames = video_frames.permute(0, 2, 1, 3, 4)
        # Extract features
        output = {}
        features = self.backbone(video_frames)
        features = features.mean(dim=[2, 3, 4])
        # Classify
        for actions_name in self.classifiers.keys():
            output[actions_name] = self.classifiers[actions_name](features)

        return output


model_factory = {
    "resnet3d": R3d,
    "stlt": Stlt,
    "swin": SwinModel,
}
