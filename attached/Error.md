
# Error 
## Training model
1. Cuda error 
```
  File "/home/linlin/ll_docker/sportsnoma-deep-learning/sports_env/lib/python3.8/site-packages/torch/autograd/__init__.py", line 197, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling `cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)`
```
- Solution: `unset LD_LIBRARY_PATH`

## Docker GPU error
```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
```
- Don't use GPU on docker temporarily
- Wait solution ??

## Pytest error 
```
The setup method ‘route’ can no longer be called on the application. It has already handled its first request, any changes will not be applied consistently. Make sure all imports, decorators, functions, etc. needed to set up the application are done before running it.
```
- solution: during pytest: need to create app fixtures
```
  @pytest.fixture
  def client():
      app.config.update({"TESTING": True})
  
      with app.test_client() as client:
          yield client
  
  def test_failure(client):
      response = client.get('/null')
```
## Check the workflow 

img /= 255.0
mean = [0.485, 0.456, 0.406] # Here it's ImageNet statistics
std = [0.229, 0.224, 0.225]

for i in range(3): # Considering an ordering NCHW (batch, channel, height, width)
    img[i, :, :] -= mean[i]
    img[i, :, :] /= std[i]

