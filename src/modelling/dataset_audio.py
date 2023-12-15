from datetime import datetime

import torch
import torchaudio.functional as F
import torchaudio.transforms as T
from yacs.config import CfgNode

from modelling.dataset_proto import ProtoDataset
from utils.data_utils import (
    compile_transforms,
    extract_audio_segments,
    get_audio_transforms,
    video2audio_indices,
)
from utils.samplers import get_sampler


class AudioDataset(ProtoDataset):
    def __init__(self, cfg: CfgNode, train: bool = False):
        self.cfg = cfg
        self.train = train
        self.dataset_type = "audio"
        self.dataset_name = (
            cfg.TRAIN_DATASET_NAME if self.train else cfg.VAL_DATASET_NAME
        )
        assert (
            self.dataset_name == "EPIC-KITCHENS"
        ), "Audio only defined for 'EPIC-KITCHENS'!"
        self.dataset_path = (
            cfg.TRAIN_DATASET_PATH if self.train else cfg.VAL_DATASET_PATH
        )
        self.create_dataset()
        # Define transform
        self.spectrogram = T.MelSpectrogram(
            n_fft=1024,
            win_length=160,
            hop_length=80,
            center=True,
            pad_mode="reflect",
            power=2.0,
        )
        self.sampler = get_sampler[self.cfg.MODEL_NAME](cfg=cfg, train=train)

    def get_difference_seconds(self, start_timestamp, stop_timestamp):
        start_timestamp = datetime.strptime(start_timestamp, "%H:%M:%S.%f")
        stop_timestamp = datetime.strptime(stop_timestamp, "%H:%M:%S.%f")
        return (stop_timestamp - start_timestamp).total_seconds()

    def get_audio_length(self, sample):
        # Returns the video length in seconds
        start_timestamp = sample["start_timestamp"]
        stop_timestamp = sample["stop_timestamp"]
        return self.get_difference_seconds(start_timestamp, stop_timestamp)

    def get_timestamp_seconds(self, timestamp):
        # Returns the timestamp total number of seconds
        timestamp = datetime.strptime(timestamp, "%H:%M:%S.%f")
        reference_timestamp = datetime.strptime("00:00:00.00", "%H:%M:%S.%f")
        return (timestamp - reference_timestamp).total_seconds()

    def get_audio_spectrograms(self, **kwargs):
        indices = kwargs.pop("indices")
        video_id = kwargs.pop("video_id")
        start_timestamp = kwargs.pop("start_timestamp")
        stop_timestamp = kwargs.pop("stop_timestamp")

        # Fix for omnivore so we don't need to re-implement method
        if "resource" in kwargs:
            resource = kwargs.pop("resource")
        else:
            resource = self.resource

        audio_sample_rate = self.cfg.AUDIO_SAMPLE_RATE
        frame_offset = int(
            self.get_timestamp_seconds(start_timestamp) * audio_sample_rate
        )
        num_frames = int(
            self.get_difference_seconds(start_timestamp, stop_timestamp)
            * audio_sample_rate
        )
        # Extract audio
        audio = torch.tensor(
            resource[video_id][frame_offset : frame_offset + num_frames]
        ).unsqueeze(0)
        if self.cfg.AUDIO_RESAMPLE_RATE:
            audio = F.resample(audio, audio_sample_rate, self.cfg.AUDIO_RESAMPLE_RATE)
            audio_sample_rate = self.cfg.AUDIO_RESAMPLE_RATE

        # Extract audio segments
        audio_indices = video2audio_indices(
            indices,
            video_fps=self.cfg.VIDEO_FPS,
            audio_sample_rate=audio_sample_rate,
        )
        segment_length = int(self.cfg.AUDIO_SEGMENT_LENGTH * audio_sample_rate)
        audio_segments = extract_audio_segments(
            audio,
            segment_length=segment_length,
            audio_indices=audio_indices,
        )
        spectrogram = self.spectrogram(
            audio_segments
        )  # [Eval_Clips x Frames, Height (Freqs), Width (Timesteps)]
        return spectrogram

    def __getitem__(self, idx: int):
        output = {self.dataset_type: []}
        if not hasattr(self, "resource"):
            self.open_resource()
        output["id"] = self.dataset[idx]["id"]
        if not hasattr(self, "indices"):
            indices = self.sampler(
                video_length=self.get_video_length(self.dataset[idx])
            )
        else:
            indices = self.indices
        output["indices"] = indices
        output["start_frame"] = int(
            self.get_timestamp_seconds(self.dataset[idx]["start_timestamp"])
            * self.cfg.VIDEO_FPS
        )
        # Check for existing transforms
        if not hasattr(self, "existing_transforms"):
            existing_transforms = {}
        else:
            existing_transforms = self.existing_transforms
        # Obtain the spectograms
        output[self.dataset_type] = self.get_audio_spectrograms(
            indices=indices,
            video_id=self.dataset[idx]["id"],
            start_timestamp=self.dataset[idx]["start_timestamp"],
            stop_timestamp=self.dataset[idx]["stop_timestamp"],
        ).unsqueeze(0)
        # Get augmentations
        audio_transforms_dict = get_audio_transforms(
            augmentations_list=self.cfg.AUGMENTATIONS.AUDIO,
            cfg=self.cfg,
            train=self.train,
            existing_transforms=existing_transforms,
        )
        self.enforced_transforms = audio_transforms_dict
        audio_transforms = compile_transforms(audio_transforms_dict)
        output[self.dataset_type] = audio_transforms(output[self.dataset_type])
        # [Eval_Clips x Frames, 1, Spatial size, Spatial size]
        output[self.dataset_type] = output[self.dataset_type].reshape(
            -1,
            self.cfg.NUM_FRAMES,
            1,
            output[self.dataset_type].shape[-2],
            output[self.dataset_type].shape[-1],
        )  # [Eval_Clips, Frames, 1, Spatial size, Spatial size]
        output[self.dataset_type] = output[self.dataset_type].repeat(1, 1, 3, 1, 1)
        # [Eval_Clips, Frames, 3, Spatial size, Spatial size]
        # Obtain video labels
        output["labels"] = self.get_actions(self.dataset[idx])
        return output
