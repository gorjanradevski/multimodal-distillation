from modelling.dataset_video import VideoDataset, LayoutDataset
from torch.utils.data import Dataset
from yacs.config import CfgNode


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


dataset_factory = {
    "video": VideoDataset,
    "layout": LayoutDataset,
    "video_layout": VideoLayoutDataset,
}
