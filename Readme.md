https://www.youtube.com/watch?v=Kzrfw-tAZew
https://github.com/abhishekkrthakur/melanoma-deep-learning


## training

- add tensorboard for training 
- improve training code 
  - improve the model structure 
Added a validation set for monitoring performance on unseen data.
Added a check to ensure that the device is set to GPU if available, otherwise use the CPU.
Added the option to resume training from a checkpoint.
Moved the training loop into a function that takes in the model, loss function, optimizer, data loaders, and other hyperparameters as arguments.
Added comments for better readability.
- overfitting 

`tensorboard --logdir=mela_api`

## flask 
- add image check 

## docker 
- run train 
- run flask
- docker
  - run it with gpu, 
  - create a new one using current script 
  - get a docker image with well-training using current script 

## Error 
### Training model
```
  File "/home/linlin/ll_docker/melanoma-deep-learning/mela_env/lib/python3.8/site-packages/torch/autograd/__init__.py", line 197, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling `cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)`
```
- Solution: `unset LD_LIBRARY_PATH`

unset LD_LIBRARY_PATH
1. url perssion
- Error: `urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired (_ssl.c:1131)>`
- Solution
```
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```
2. get error: 
```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
```

- solution: reinstall docker, refer: [install docker on ubuntu](intallation.md)




