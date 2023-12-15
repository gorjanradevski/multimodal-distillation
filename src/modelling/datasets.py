from torch.utils.data import Dataset
from yacs.config import CfgNode

from modelling.dataset_audio import AudioDataset
from modelling.dataset_layout import LayoutDataset
from modelling.dataset_video import VideoDataset


class VideoFlow(Dataset):
    # FIXME: Hacky
    def __init__(self, cfg: CfgNode, train: bool = False):
        # Flow dataset
        flow_cfg = cfg.clone()
        flow_cfg.defrost()
        flow_cfg.DATASET_TYPE = "flow"
        flow_cfg.freeze()
        self.flow_dataset = VideoDataset(flow_cfg, train=train)
        # video dataset
        video_cfg = cfg.clone()
        video_cfg.defrost()
        video_cfg.DATASET_TYPE = "video"
        video_cfg.freeze()
        self.video_dataset = VideoDataset(video_cfg, train=train)

    def __len__(self):
        return self.flow_dataset.__len__()

    def __getitem__(self, idx: int):
        flow_dict = self.flow_dataset[idx]
        self.video_dataset.set_indices(flow_dict["indices"])
        self.video_dataset.set_existing_transforms(
            self.flow_dataset.enforced_transforms
        )
        video_dict = self.video_dataset[idx]
        # Gather from both dicts
        output = {}
        for c_dict in [flow_dict, video_dict]:
            for key in c_dict.keys():
                output[key] = c_dict[key]

        return output


class VideoAudio(Dataset):
    # FIXME: Hacky
    def __init__(self, cfg: CfgNode, train: bool = False):
        # Audio dataset
        audio_cfg = cfg.clone()
        audio_cfg.defrost()
        audio_cfg.DATASET_TYPE = "audio"
        audio_cfg.freeze()
        self.audio_dataset = AudioDataset(audio_cfg, train=train)
        # video dataset
        video_cfg = cfg.clone()
        video_cfg.defrost()
        video_cfg.DATASET_TYPE = "video"
        video_cfg.freeze()
        self.video_dataset = VideoDataset(video_cfg, train=train)

    def __len__(self):
        return self.audio_dataset.__len__()

    def __getitem__(self, idx: int):
        audio_dict = self.audio_dataset[idx]
        self.video_dataset.set_indices(audio_dict["indices"])
        video_dict = self.video_dataset[idx]
        # Gather from both dicts
        output = {}
        for c_dict in [audio_dict, video_dict]:
            for key in c_dict.keys():
                output[key] = c_dict[key]

        return output


class VideoFlowAudio(Dataset):
    # FIXME: Hacky
    def __init__(self, cfg: CfgNode, train: bool = False):
        # Audio dataset
        audio_cfg = cfg.clone()
        audio_cfg.defrost()
        audio_cfg.DATASET_TYPE = "audio"
        audio_cfg.freeze()
        self.audio_dataset = AudioDataset(audio_cfg, train=train)
        # Flow dataset
        flow_cfg = cfg.clone()
        flow_cfg.defrost()
        flow_cfg.DATASET_TYPE = "flow"
        flow_cfg.freeze()
        self.flow_dataset = VideoDataset(flow_cfg, train=train)
        # video dataset
        video_cfg = cfg.clone()
        video_cfg.defrost()
        video_cfg.DATASET_TYPE = "video"
        video_cfg.freeze()
        self.video_dataset = VideoDataset(video_cfg, train=train)

    def set_weighted(self):
        from copy import deepcopy

        copy_self = deepcopy(self)
        copy_self.audio_dataset = self.audio_dataset.set_weighted()
        copy_self.video_dataset = self.video_dataset.set_weighted()
        copy_self.flow_dataset = self.flow_dataset.set_weighted()

        return copy_self

    def __len__(self):
        return self.flow_dataset.__len__()

    def __getitem__(self, idx: int):
        audio_dict = self.audio_dataset[idx]
        self.flow_dataset.set_indices(audio_dict["indices"])
        flow_dict = self.flow_dataset[idx]
        self.video_dataset.set_indices(audio_dict["indices"])
        self.video_dataset.set_existing_transforms(
            self.flow_dataset.enforced_transforms
        )
        video_dict = self.video_dataset[idx]
        # Gather from both dicts
        output = {}
        for c_dict in [flow_dict, video_dict, audio_dict]:
            for key in c_dict.keys():
                output[key] = c_dict[key]

        return output


class VideoLayoutDataset(Dataset):
    # FIXME: Hacky
    def __init__(self, cfg: CfgNode, train: bool = False):
        self.layout_dataset = LayoutDataset(cfg, train=train)
        video_cfg = cfg.clone()
        video_cfg.defrost()
        video_cfg.DATASET_TYPE = "video"
        video_cfg.freeze()
        self.video_dataset = VideoDataset(video_cfg, train=train)

    def __len__(self):
        return self.layout_dataset.__len__()

    def __getitem__(self, idx: int):
        layout_dict = self.layout_dataset[idx]
        self.video_dataset.set_indices(layout_dict["indices"])
        video_dict = self.video_dataset[idx]
        # Gather from both dicts
        output = {}
        for c_dict in [layout_dict, video_dict]:
            for key in c_dict.keys():
                output[key] = c_dict[key]

        return output


class VideoLayoutFlow(Dataset):
    def __init__(self, cfg: CfgNode, train: bool = False):
        # FIXME: Hacky
        # Layout dataset
        self.layout_dataset = LayoutDataset(cfg, train=train)
        # Flow dataset
        flow_cfg = cfg.clone()
        flow_cfg.defrost()
        flow_cfg.DATASET_TYPE = "flow"
        flow_cfg.freeze()
        self.flow_dataset = VideoDataset(flow_cfg, train=train)
        # video dataset
        video_cfg = cfg.clone()
        video_cfg.defrost()
        video_cfg.DATASET_TYPE = "video"
        video_cfg.freeze()
        self.video_dataset = VideoDataset(video_cfg, train=train)

    def __len__(self):
        return self.flow_dataset.__len__()

    def __getitem__(self, idx: int):
        flow_dict = self.flow_dataset[idx]
        self.video_dataset.set_indices(flow_dict["indices"])
        self.video_dataset.set_existing_transforms(
            self.flow_dataset.enforced_transforms
        )
        video_dict = self.video_dataset[idx]
        self.layout_dataset.set_indices(flow_dict["indices"])
        layout_dict = self.layout_dataset[idx]
        # Gather from both dicts
        output = {}
        for c_dict in [flow_dict, video_dict, layout_dict]:
            for key in c_dict.keys():
                output[key] = c_dict[key]

        return output


dataset_factory = {
    "video": VideoDataset,
    "flow": VideoDataset,
    "depth": VideoDataset,
    "layout": LayoutDataset,
    "video_layout": VideoLayoutDataset,
    "video_flow": VideoFlow,
    "video_layout_flow": VideoLayoutFlow,
    "video_flow_audio": VideoFlowAudio,
    "audio": AudioDataset,
    "video_audio": VideoAudio,
}
