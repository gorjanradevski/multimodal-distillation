# Multimodal Distillation for Egocentric Action Recognition

This repository contains the implementation of the paper [Multimodal Distillation for Egocentric Action Recognition](https://arxiv.org/abs/2307.07483).

### Preparing Epic-Kitchens

Download the RGB frames from [here](https://github.com/epic-kitchens/epic-kitchens-download-scripts) and pack them as an HDF5 file. Detailed instructions will follow soon.

### Inference on Epic-Kitchens

1. Download our model from [here](https://drive.google.com/drive/folders/1KUBiwGodTLqtuoRoxJVbpwBZ8uieJCtm?usp=sharing), and place it `experiments/`.
2. Run inference as indicated bellow:

```python
python src/inference.py --experiment_path "experiments/epic-kitchens-swint-distill-flow-audio" --opts DATASET_TYPE "video"
```

### Inference on Something-Something

Instructions will follow :)

### Citations

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