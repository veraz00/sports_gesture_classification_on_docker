# Sports Gesture Classification on Docker

## Overview 
- This is an API for sports gesture image classification
- Features
  - The model use **Swin Transformer** (accuracy = 80% on validation dataset)
  - Run training and flask API on **Docker container**
  - Provide `onnx` (CPU device) 

- Demo
https://youtu.be/2DWbemtVgis

## Run 
- Build a docker based on `DockerFile`
- Train: `bash train.sh` 
- Flask api: `bash api.sh`
    - Browser: `http://localhost:12000/`


## Dataset 
[Kaggle Sports Gesture](https://www.kaggle.com/datasets/gpiosenka/sports-classification)
- Classes: 100
- Size: 224 X224 X 3
- Format: `.jpg` 
- Number of images: 14572 (train: 4572 , valid: 500, test: 500)

## Train
The training code is from [Kaggle Sports Gesture Competition: SwinTransformer from Timm](https://www.kaggle.com/code/pkbpkb0055/99-2-classification-using-swin-transformer)
```
python3 train.py --config {config_file}
```
- Check the process on Tensorboard: `tensorboard --logdir = sports_api`



## Run on Docker 
[Steps about Run on Docker](attached/Build_docker_image.md)



## Refer
1. Inspired from sportsnoma classification: https://www.youtube.com/watch?v=Kzrfw-tAZew
https://github.com/abhishekkrthakur/sportsnoma-deep-learning
2. [Kaggle: Sports Gesture Classification](https://www.kaggle.com/datasets/gpiosenka/sports-classification)
3. [Kaggle Sports Gesture Competition: SwinTransformer from Timm](https://www.kaggle.com/code/pkbpkb0055/99-2-classification-using-swin-transformer)
4. [Pytest on Flask API](https://github.com/haythemtellili/NLP-Multilabel-classification/blob/main/tests/test_model.py)