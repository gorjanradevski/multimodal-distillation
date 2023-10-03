import random

import numpy as np
from yacs.config import CfgNode


class StltSampler:
    def __init__(self, cfg: CfgNode, train: bool = False):
        self.cfg = cfg
        self.train = train

    def __call__(self, video_length: int):
        seg_size = float(video_length - 1) / self.cfg.NUM_FRAMES
        seq = []
        for i in range(self.cfg.NUM_FRAMES):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            if self.train:
                seq.append(random.randint(start, end))
            else:
                seq.append((start + end) // 2)

        seq = np.array(seq)

        return seq


# Adapted from https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/mmaction/datasets/pipelines/loading.py#L79
class GeneralSampler:
    def __init__(self, cfg: CfgNode, train: bool = False):
        self.cfg = cfg
        self.clip_len = self.cfg.NUM_FRAMES
        self.frame_interval = self.cfg.STRIDE
        self.num_clips = self.cfg.NUM_TEST_CLIPS
        self.temporal_jitter = False
        self.twice_sample = False
        self.out_of_bound_opt = "repeat_last"
        self.train = train
        self.frame_uniform = cfg.FRAME_UNIFORM
        assert self.out_of_bound_opt in ["loop", "repeat_last"]

    def _get_train_clips(self, video_length):
        """Get clip offsets in train mode.
        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.
        Args:
            video_length (int): Total number of frame in the video.
        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (video_length - ori_clip_len + 1) // self.num_clips

        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips
            )
        elif video_length > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(video_length - ori_clip_len + 1, size=self.num_clips)
            )
        elif avg_interval == 0:
            ratio = (video_length - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=int)

        return clip_offsets

    def _get_test_clips(self, video_length):
        """Get clip offsets in test mode.
        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2. If set twice_sample True, it will sample
        frames together without fixed shift. If the total number of frames is
        not enough, it will return all zero indices.
        Args:
            video_length (int): Total number of frame in the video.
        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (video_length - ori_clip_len + 1) / float(self.num_clips)
        if video_length > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(int)
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=int)
        return clip_offsets

    def _sample_clips(self, video_length):
        """Choose clip offsets for the video in a given mode.
        Args:
            video_length (int): Total number of frame in the video.
        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.train:
            clip_offsets = self._get_train_clips(video_length)
        else:
            clip_offsets = self._get_test_clips(video_length)

        return clip_offsets

    def get_seq_frames(self, video_length):
        """
        Modified from https://github.com/facebookresearch/SlowFast/blob/64abcc90ccfdcbb11cf91d6e525bed60e92a8796/slowfast/datasets/ssv2.py#L159
        Given the video index, return the list of sampled frame indexes.
        Args:
            video_length (int): Total number of frame in the video.
        Returns:
            seq (list): the indexes of frames of sampled from the video.
        """
        # Update desired clip_len based on num_clips
        clip_len = self.clip_len * self.num_clips
        # Proceed to rest
        seg_size = float(video_length - 1) / clip_len
        seq = []
        for i in range(clip_len):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            if self.train:
                seq.append(random.randint(start, end))
            else:
                seq.append((start + end) // 2)

        return np.array(seq)

    def __call__(self, video_length: int):
        """Perform the SampleFrames loading.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.frame_uniform:  # sthv2 sampling strategy
            frame_inds = self.get_seq_frames(video_length)
        else:
            clip_offsets = self._sample_clips(video_length)
            frame_inds = (
                clip_offsets[:, None]
                + np.arange(self.clip_len)[None, :] * self.frame_interval
            )
            frame_inds = np.concatenate(frame_inds)

            if self.temporal_jitter:
                perframe_offsets = np.random.randint(
                    self.frame_interval, size=len(frame_inds)
                )
                frame_inds += perframe_offsets

            frame_inds = frame_inds.reshape((-1, self.clip_len))
            if self.out_of_bound_opt == "loop":
                frame_inds = np.mod(frame_inds, video_length)
            elif self.out_of_bound_opt == "repeat_last":
                safe_inds = frame_inds < video_length
                unsafe_inds = 1 - safe_inds
                last_ind = np.max(safe_inds * frame_inds, axis=1)
                new_inds = safe_inds * frame_inds + (unsafe_inds.T * last_ind).T
                frame_inds = new_inds
            else:
                raise ValueError("Illegal out_of_bound option.")

            frame_inds = np.concatenate(frame_inds)

        return frame_inds.astype(int)


# TODO: Find better naming for these
get_sampler = {
    "swin": GeneralSampler,
    "stlt": StltSampler,
    "resnet3d": GeneralSampler,
}
