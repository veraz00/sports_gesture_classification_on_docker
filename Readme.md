# Sports Gesture Classification on Docker

## Overview 
- This is an API for sports gesture image classification
- Features
  - The model use **Swin Transformer** (accuracy = 80% on validation dataset)
  - Run training and flask API on **Docker container**
  - Provide `onnx` (CPU device) and `pt` (GPU device) to fasten the prediction time 
    | weight | prediction time on cpu (ms) | prediction time on gpu (ms) |
    | --| -- | -- |
    | pt | 0.1821 | 0.0219 |
    | onnx | 0.1062 | 0.0386 |

- Demo
https://youtu.be/2DWbemtVgis


## Dataset 
[Kaggle Sports Gesture](https://www.kaggle.com/datasets/gpiosenka/sports-classification)
- Classes: 100
- Size: 224 X224 X 3
- Format: `.jpg` 
- Number of images: 14572 (train: 4572 , valid: 500, test: 500)

## Train
The training code is from [Kaggle Sports Gesture Competition: SwinTransformer from Timm](https://www.kaggle.com/code/pkbpkb0055/99-2-classification-using-swin-transformer)
```
python3 train.py
```
- Check the process on Tensorboard: `tensorboard --logdir = sports_api`

## Add Flask
```
python3 api.py
```
- Browser: `http://localhost:12000/`



## Run on Docker 
[Steps about Run on Docker](attached/Build_docker_image.md)


## Error 
### Training model
1. Cuda error 
```
  File "/home/linlin/ll_docker/sportsnoma-deep-learning/sports_env/lib/python3.8/site-packages/torch/autograd/__init__.py", line 197, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling `cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)`
```
- Solution: `unset LD_LIBRARY_PATH`

2. Docker GPU error
```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
```
- Don't use GPU on docker temporarily
- Wait solution ??


## Refer
1. Inspired from sportsnoma classification: https://www.youtube.com/watch?v=Kzrfw-tAZew
https://github.com/abhishekkrthakur/sportsnoma-deep-learning
2. [Kaggle: Sports Gesture Classification](https://www.kaggle.com/datasets/gpiosenka/sports-classification)
3. [Kaggle Sports Gesture Competition: SwinTransformer from Timm](https://www.kaggle.com/code/pkbpkb0055/99-2-classification-using-swin-transformer)