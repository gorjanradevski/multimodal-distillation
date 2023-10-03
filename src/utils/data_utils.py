import math
import random
from copy import deepcopy
from typing import List, Tuple

import ffmpeg
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as TAT
from PIL import Image
from torch.nn.modules.utils import _pair
from torchaudio.transforms import AmplitudeToDB
from torchvision.transforms import ColorJitter, Compose, Normalize, RandomCrop, Resize
from torchvision.transforms import functional as TF
from yacs.config import CfgNode

spatial_sizes = {
    "resnet3d": 224,  # HACK, because we resize inside the model to 112
    "video_mae": 224,
    "swin": 224,
}


def load_video(in_filepath: str):
    """Loads a video from a filepath."""
    probe = ffmpeg.probe(in_filepath)
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
        None,
    )
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    out, _ = (
        ffmpeg.input(in_filepath)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        # https://github.com/kkroening/ffmpeg-python/issues/68#issuecomment-443752014
        .global_args("-loglevel", "error")
        .run(capture_stdout=True)
    )
    video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])

    return video


class IdentityTransform:
    def __init__(self, **kwargs):
        pass

    def __call__(self, img):
        return img


class ToTensor:
    def __init__(self, **kwargs):
        pass

    def __call__(self, img):
        return TF.to_tensor(img)


class VideoNormalize:
    def __init__(self, model_name: str, **kwargs):
        normalizations = {
            # Orig: mean: (123.675, 116.28, 103.53);
            # std: (58.395, 57.12, 57.375), we divide by
            # 255 because our images are normalized by 255
            "swin": {"mean": (0.4850, 0.4560, 0.4060), "std": (0.2290, 0.2240, 0.2250)},
            "resnet3d": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
            # https://rwightman.github.io/pytorch-image-models/models/vision-transformer/
            "video_mae": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
        }
        self.normalize = Normalize(
            mean=normalizations[model_name]["mean"],
            std=normalizations[model_name]["std"],
        )

    def __call__(self, img: torch.Tensor):
        return self.normalize(img)


class VideoColorJitter:
    # Adapted from: https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py#L1140
    def __init__(self, **kwargs):
        (
            self.fn_idx,
            self.brightness_factor,
            self.contrast_factor,
            self.saturation_factor,
            self.hue_factor,
        ) = ColorJitter.get_params(
            brightness=(0.75, 1.25),
            contrast=(0.75, 1.25),
            saturation=(0.75, 1.25),
            hue=(-0.1, 0.1),
        )

    def __call__(self, img: Image):
        for fn_id in self.fn_idx:
            if fn_id == 0 and self.brightness_factor is not None:
                img = TF.adjust_brightness(img, self.brightness_factor)
            elif fn_id == 1 and self.contrast_factor is not None:
                img = TF.adjust_contrast(img, self.contrast_factor)
            elif fn_id == 2 and self.saturation_factor is not None:
                img = TF.adjust_saturation(img, self.saturation_factor)
            elif fn_id == 3 and self.hue_factor is not None:
                img = TF.adjust_hue(img, self.hue_factor)

        return img


class VideoRandomHorizontalFlip:
    def __init__(self, p: float, **kwargs):
        self.flip = torch.rand(1) < p

    def __call__(self, img):
        if self.flip:
            img = TF.hflip(img)
        return img


class VideoRandomCrop:
    def __init__(self, spatial_size: int, **kwargs):
        self.spatial_size = (spatial_size, spatial_size)

    def __call__(self, frame: Image):
        if (
            not hasattr(self, "top")
            and not hasattr(self, "left")
            and not hasattr(self, "height")
            and not hasattr(self, "width")
        ):
            self.top, self.left, self.height, self.width = RandomCrop.get_params(
                frame, self.spatial_size
            )

        return TF.crop(frame, self.top, self.left, self.height, self.width)


class VideoResize:
    def __init__(self, spatial_size: int, train: bool = False, **kwargs):
        frame_size = (
            int(spatial_size * random.uniform(1.15, 1.43)) if train else spatial_size
        )
        self.resize = Resize(frame_size, antialias=True)

    def __call__(self, img: Image):
        return self.resize(img)


class VideoInferenceCrop:
    def __init__(self, spatial_size: int, num_test_crops: int, **kwargs):
        self.spatial_size = spatial_size
        self.num_test_crops = num_test_crops
        assert self.num_test_crops in [1, 3], "Can be only 1 or 3"

    def __call__(self, img: torch.Tensor):
        if self.num_test_crops == 1:
            return TF.center_crop(img, self.spatial_size)
        # Three-crop inference. New dimension prepended and images are stacked along it.
        crop_size = _pair(self.spatial_size)
        img_h, img_w = img.shape[-2:]
        crop_w, crop_h = crop_size
        assert crop_h == img_h or crop_w == img_w

        if crop_h == img_h:
            w_step = (img_w - crop_w) // 2
            offsets = [
                (0, 0),  # left
                (2 * w_step, 0),  # right
                (w_step, 0),  # middle
            ]
        elif crop_w == img_w:
            h_step = (img_h - crop_h) // 2
            offsets = [
                (0, 0),  # top
                (0, 2 * h_step),  # down
                (0, h_step),  # middle
            ]

        img_cropped = []
        for x_offset, y_offset in offsets:
            crop = img[..., y_offset : y_offset + crop_h, x_offset : x_offset + crop_w]
            img_cropped.append(crop)
        img_cropped = torch.stack(img_cropped, dim=0)
        return img_cropped


class AudioResize:
    def __init__(self, spatial_size, **kwargs):
        self.spatial_size = spatial_size

    def __call__(self, img: torch.tensor):
        return F.interpolate(img, size=self.spatial_size)


class AudioTimeStretch:
    def __init__(
        self, p: float = 0.5, spatial_size: int = 224, train: bool = False, **kwargs
    ):
        self.spatial_size = spatial_size
        self.train = train
        self.p = p
        self.transform = TAT.TimeStretch() if self.train else IdentityTransform()

    def __call__(self, img: Image):
        rate = random.uniform(0.85, 1.15)
        if random.uniform(0, 1) < self.p:
            return self.transform(img, rate)
        return img


class AudioTimeMasking:
    def __init__(
        self, p: float = 0.5, time_mask_param: int = 80, train: bool = False, **kwargs
    ):
        self.train = train
        self.p = p
        self.transform = (
            TAT.TimeMasking(time_mask_param=time_mask_param, iid_masks=True)
            if self.train
            else IdentityTransform()
        )

    def __call__(self, img: torch.Tensor):
        if random.uniform(0, 1) < self.p:
            return self.transform(img)
        return img


class AudioFrequencyMasking:
    def __init__(
        self, p: float = 0.5, freq_mask_param: int = 80, train: bool = False, **kwargs
    ):
        self.train = train
        self.p = p
        self.transform = (
            TAT.FrequencyMasking(freq_mask_param=freq_mask_param, iid_masks=True)
            if self.train
            else IdentityTransform()
        )

    def __call__(self, img: torch.Tensor):
        if random.uniform(0, 1) < self.p:
            return self.transform(img)
        return img


class AudioNormalize:
    def __init__(self, **kwargs):
        self.mean = -28.1125
        self.std = 16.5627

    def __call__(self, audio: torch.Tensor):
        return (audio - self.mean) / self.std


class AudioAmplitudeToDB:
    def __init__(self, **kwargs):
        self.amplitude_to_db = AmplitudeToDB()

    def __call__(self, audio: torch.Tensor):
        return self.amplitude_to_db(audio)


def compile_transforms(transforms):
    return Compose(transforms.values())


augname2aug = {
    "AudioTimeStretch": AudioTimeStretch,
    "AudioTimeMasking": AudioTimeMasking,
    "AudioFrequencyMasking": AudioFrequencyMasking,
    "AudioResize": AudioResize,
    "AudioAmplitudeToDB": AudioAmplitudeToDB,
    "AudioNormalize": AudioNormalize,
    "IdentityTransform": IdentityTransform,
    "VideoColorJitter": VideoColorJitter,
    "VideoRandomHorizontalFlip": VideoRandomHorizontalFlip,
    "VideoRandomCrop": VideoRandomCrop,
    "VideoInferenceCrop": VideoInferenceCrop,
    "VideoResize": VideoResize,
    "ToTensor": ToTensor,
    "VideoNormalize": VideoNormalize,
}


def get_video_transforms(
    augmentations_list: List[str], cfg: CfgNode, train: bool = False, **kwargs
):
    aug_list = deepcopy(augmentations_list)
    existing_transforms = kwargs.pop("existing_transforms", {})
    # Testing
    if not train:
        return {
            "VideoResize": VideoResize(
                spatial_size=spatial_sizes[cfg.MODEL_NAME], train=False
            ),
            "ToTensor": ToTensor(),
            "VideoInferenceCrop": VideoInferenceCrop(
                spatial_size=spatial_sizes[cfg.MODEL_NAME],
                num_test_crops=cfg.NUM_TEST_CROPS,
            ),
        }
    # During training, always add ToTensor
    aug_list.append("ToTensor")

    return {
        aug_name: augname2aug[aug_name](
            spatial_size=spatial_sizes[cfg.MODEL_NAME],
            train=train,
            p=0.5,
            model_name=cfg.MODEL_NAME,
            **kwargs,
        )
        if aug_name not in existing_transforms
        else existing_transforms[aug_name]
        for aug_name in aug_list
    }


def get_audio_transforms(
    augmentations_list: List[str], cfg: CfgNode, train: bool = False, **kwargs
):
    aug_list = deepcopy(augmentations_list)
    existing_transforms = kwargs.pop("existing_transforms", {})
    spatial_size = spatial_sizes[cfg.MODEL_NAME]
    # Testing
    if not train:
        return {
            "AudioResize": AudioResize(
                spatial_size=(spatial_size, spatial_size), train=False
            ),
            "AudioAmplitudeToDB": AudioAmplitudeToDB(),
        }
    # During training, always add AudioNormalize to the list
    aug_list.append("AudioAmplitudeToDB")
    return {
        aug_name: augname2aug[aug_name](
            spatial_size=spatial_size, train=train, **kwargs
        )
        if aug_name not in existing_transforms
        else existing_transforms[aug_name]
        for aug_name in aug_list
    }


def get_normalizer(input_type: str, model_name: str):
    if input_type == "audio":
        return AudioNormalize(model_name=model_name)
    elif input_type == "video" or input_type == "flow":
        return VideoNormalize(model_name=model_name)

    raise ValueError(f"{input_type} not recognized!")


def video2audio_indices(indices, video_fps, audio_sample_rate):
    return [int(index * audio_sample_rate / video_fps) for index in indices]


def audio2video_indices(indices, video_fps, audio_sample_rate):
    return [int(index * video_fps / audio_sample_rate) for index in indices]


def extract_audio_segments(audio_frames, segment_length, audio_indices):
    _, num_frames = audio_frames.shape

    audio_segments = []
    for audio_index in audio_indices:
        centre_frame = audio_index
        left_frame = centre_frame - math.floor(segment_length / 2)
        right_frame = centre_frame + math.ceil(segment_length / 2)
        if left_frame < 0 and right_frame > num_frames:
            samples = torch.nn.functional.pad(
                audio_frames, pad=(abs(left_frame), right_frame - num_frames)
            )
        elif left_frame < 0:
            samples = torch.nn.functional.pad(audio_frames, pad=(abs(left_frame), 0))[
                :, :segment_length
            ]
        elif right_frame > num_frames:
            samples = torch.nn.functional.pad(
                audio_frames, pad=(0, right_frame - num_frames)
            )[:, -segment_length:]
        else:
            samples = audio_frames[:, left_frame:right_frame]
        audio_segments.append(samples)
    audio_segments = torch.cat(audio_segments, dim=0)
    return audio_segments


def fix_box(box: List[int], video_size: Tuple[int, int]):
    # Cast box elements to integers
    box = [max(0, int(b)) for b in box]
    # If x1 > x2 or y1 > y2 switch (Hack)
    if box[0] > box[2]:
        box[0], box[2] = box[2], box[0]
    if box[1] > box[3]:
        box[1], box[3] = box[3], box[1]
    # Clamp to max size (Hack)
    if box[0] >= video_size[1]:
        box[0] = video_size[1] - 1
    if box[1] >= video_size[0]:
        box[1] = video_size[0] - 1
    if box[2] >= video_size[1]:
        box[2] = video_size[1] - 1
    if box[3] >= video_size[0]:
        box[3] = video_size[0] - 1
    # Fix if equal (Hack)
    if box[0] == box[2] and box[0] == 0:
        box[2] = 1
    if box[1] == box[3] and box[1] == 0:
        box[3] = 1
    if box[0] == box[2]:
        box[0] -= 1
    if box[1] == box[3]:
        box[1] -= 1
    return box


separator = "=" * 40


def rgb_to_flow_index(rgb_index: int, stride: int = 1):
    # https://github.com/epic-kitchens/epic-kitchens-download-scripts/issues/17#issuecomment-1222288006
    # https://github.com/epic-kitchens/epic-kitchens-55-lib/blob/7f2499aff5fdb62a66e6da92322e5c060ea4a414/epic_kitchens/video.py#L67
    # https://github.com/epic-kitchens/C1-Action-Recognition-TSN-TRN-TSM/blob/master/src/convert_rgb_to_flow_frame_idxs.py#L24
    return int(np.ceil(rgb_index / stride))
