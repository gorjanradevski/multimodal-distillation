# Multimodal Distillation for Egocentric Action Recognition

This repository contains the implementation of the paper [Multimodal Distillation for Egocentric Action Recognition](https://arxiv.org/abs/2307.07483).

![Teaser](data/assets/teaser.png)

## Reproducing the virtual environment

The main dependencies that you need to install to reproduce the virtual environment are [PyTorch](https://pytorch.org/), and:

```shell
pip install accelerate tqdm h5py yacs timm einops
```

## Preparing Epic-Kitchens and Something-Something

We store all data (video frames, optical flow frames, audios, etc.) is an efficient HDF5 file where each video represents a dataset within the HDF5 file, and the n-th element of the dataset contains the bytes for the n-th frame of the video. Since these files are large, drop us an email and we can give you access to them.

## Inference on Epic-Kitchens

1. Download our model from [here](https://drive.google.com/drive/folders/1KUBiwGodTLqtuoRoxJVbpwBZ8uieJCtm?usp=sharing), and place it in `experiments/`.
2. Run inference as indicated bellow:

```python
python src/inference.py --experiment_path "experiments/epic-kitchens-swint-distill-flow-audio" --opts DATASET_TYPE "video"
```

## Inference on Something-Something and Something-Else

1. Download our Something-Else distilled model from [here](https://drive.google.com/drive/folders/1zK_dEGJP21xtgrZgc_gOPBPvWg-DKL2j?usp=sharing), and place it in `experiments/`.
2. Run inference as indicated bellow:

```python
python src/inference.py --experiment_path "experiments/somethiing-else-swint-distill-layout-flow" --opts DATASET_TYPE "video"
```

## Distilling from Multimodal Teachers

To reproduce the experiments (i.e., using the identical hyperparameters, where only the random seed will vary) you can do:

```python
python src/patient_distill.py --config "experiments/somethiing-else-swint-distill-layout-flow/config.yaml" --opts EXPERIMENT_PATH "experiments/experiments/reproducing-an-experiment"
```

note that this assumes access to the datasets for all modalities (video, optical flow, audio, object detections), as well as the individual (unimodal) models which constitute the multimodal ensemble teacher.

## Citation

If you find our code useful for your own research, please use the following BibTeX entry:

```tex
@inproceedings{radevski2023multimodal,
  title={Multimodal Distillation for Egocentric Action Recognition},
  author={Radevski, Gorjan and Grujicic, Dusan and Blaschko, Matthew and Moens, Marie-Francine and Tuytelaars, Tinne},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5213--5224},
  year={2023}
}
```