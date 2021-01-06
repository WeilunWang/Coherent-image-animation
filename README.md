# Coherent Image Animation using Spatial Temporal Correspondence
This repository contains the source code for the paper "Coherent Image Animation using Spatial Temporal Correspondence".

## Demo
Please refer to
```
./multimedia/demo.mp4
```

## Installation
We support ```python3```. To install the dependencies run:
```
pip install -r requirements.txt
```

### Evaluation on video reconstruction
To evaluate our model on Tai-Chi-HD dataset, run:
```
CUDA_VISIBLE_DEVICES=0 python3 run.py --config config/taichi-adv-256.yaml --checkpoint ./checkpoints/taichiHD/latest-checkpoint.pth.tar --mode reconstruction
```
