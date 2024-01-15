# This script packs a dataset of video frames into a HDF5 file. The script assumes that
# the video frames are stored in an image format (.png, .jpg, etc.) and that the frames
# for a single video are stored in a directory named after the video id.

import argparse
import os

import h5py
import numpy as np
from natsort import natsorted
from tqdm import tqdm


def pack_video_frames_to_hdf5(args):
    with h5py.File(args.save_hdf5_path, "a") as hdf5_file:
        # Iterate over all video frames
        for video_id in tqdm(natsorted(os.listdir(args.all_video_frames_path))):
            video_frames_path = os.path.join(args.all_video_frames_path, video_id)
            # Iterate over all frames in a video
            frames = []
            for frame_name in os.listdir(video_frames_path):
                frame_path = os.path.join(video_frames_path, frame_name)
                # Load frames
                with open(frame_path, "rb") as img_f:
                    binary_data = img_f.read()
                # Covert to numpy as aggregate
                frames.append(np.asarray(binary_data))
            frames = np.concatenate(frames, axis=0)
            hdf5_file.create_dataset(video_id, data=frames)


def main():
    parser = argparse.ArgumentParser(description="Pack video frames in HDF5.")
    parser.add_argument(
        "--all_video_frames_path",
        type=str,
        default="data/extracted_videos",
        help="From where to load the video frames.",
    )
    parser.add_argument(
        "--save_hdf5_path",
        type=str,
        default="data/dataset.hdf5",
        help="Where to save the HDF5 file.",
    )
    args = parser.parse_args()
    pack_video_frames_to_hdf5(args)


if __name__ == "__main__":
    main()
