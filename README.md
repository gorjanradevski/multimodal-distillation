# Multimodal Distillation for Egocentric Action Recognition

This repository contains the implementation of the paper [Multimodal Distillation for Egocentric Action Recognition](https://arxiv.org/abs/2307.07483), published at ICCV 2023.

![Teaser](data/assets/teaser.png)

## Reproducing the virtual environment

The main dependencies that you need to install to reproduce the virtual environment are [PyTorch](https://pytorch.org/), and:

```shell
pip install accelerate tqdm h5py yacs timm einops
```

## Downloading the pre-trained Swin-T model

Create a directory `./data/pretrained-backbones/` and download Swin-T from [here](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth):

```bash
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth  -O ./data/pretrained-backbones/
``` 

## Preparing Epic-Kitchens and Something-Something

We store all data (video frames, optical flow frames, audios, etc.) is an efficient HDF5 file where each video represents a dataset within the HDF5 file, and the n-th element of the dataset contains the bytes for the n-th frame of the video. Since these files are large, drop us an email and we can give you access to them.

Ones we send you the datasets, you can place them inside `./data/` - `./data/something-something/` and `./data/EPIC-KITCHENS`. Please feel free to store the data wherever you see fit, just do not forget to modify the `config.yaml` files with the appropriate location. In this README.md, we assume that all data is placed inside `./data/` and all experiments are placed inside `./experiments/`. 

## Inference on Epic-Kitchens

1. Download our Epic-Kitchens distilled model from [here](https://drive.google.com/drive/folders/1KUBiwGodTLqtuoRoxJVbpwBZ8uieJCtm?usp=sharing), and place it in `./experiments/`.
2. Run inference as indicated bellow:

```python
python src/inference.py --experiment_path "experiments/epic-kitchens-swint-distill-flow-audio" --opts DATASET_TYPE "video"
```

## Inference on Something-Something & Something-Else

1. Download our Something-Else distilled model from [here](https://drive.google.com/drive/folders/1zK_dEGJP21xtgrZgc_gOPBPvWg-DKL2j?usp=sharing) or the Something-Something distilled model from [here](https://drive.google.com/drive/folders/1qAZTKYjt-D2Y95BlTsjw2Ny9YWMWGnQC?usp=sharing), and place it in `./experiments/`.
2. Run inference as indicated bellow:

```python
python src/inference.py --experiment_path "experiments/something-swint-distill-layout-flow" --opts DATASET_TYPE "video"
```

for Something-Something, and 

```python
python src/inference.py --experiment_path "experiments/something-else-swint-distill-layout-flow" --opts DATASET_TYPE "video"
```

for Something-Else.

## Distilling from Multimodal Teachers

To reproduce the experiments (i.e., using the identical hyperparameters, where only the random seed will vary) you can do:

```python
python src/patient_distill.py --config "experiments/something-else-swint-distill-layout-flow/config.yaml" --opts EXPERIMENT_PATH "experiments/experiments/reproducing-the-something-else-experiment"
```

note that this assumes access to the datasets for all modalities (video, optical flow, audio, object detections), as well as the individual (unimodal) models which constitute the multimodal ensemble teacher.

### Model ZOO for unimodal models

- Something-Else: [Video](https://drive.google.com/drive/folders/1jNO-OBb6rmA2Gl0MS5x-gZmkG7lN5ogM?usp=sharing); [Object detections](https://drive.google.com/drive/folders/16hna0e1RnzcQ750FAD213tm5clD5e4X3?usp=sharing); [Optical Flow](https://drive.google.com/drive/folders/1GSt-ZbAoVvGI8JWIXMaOphQm4FMWlbLe?usp=sharing).
- Something-Something: [Video](#); [Object detections](#); [Optical Flow](#).
- Epic-Kitchens: [Video](https://drive.google.com/drive/folders/101kHwBAQTDbL8IpaODxZ8jG_aMurCioW?usp=sharing); [Audio](https://drive.google.com/drive/folders/1yS9apZIUPHWUpQiXi0bCHXb5sfFsvY84?usp=sharing), [Optical Flow](https://drive.google.com/drive/folders/1DBmfQo5-8AmRmrqt9ZmFL7B4EjoGLQsA?usp=sharing).

## TODOs

- [ ] Release Something-Something pretrained teachers for each modality.
- [ ] Test the codebase.
- [ ] Structure the Model ZOO part of the codebase.

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