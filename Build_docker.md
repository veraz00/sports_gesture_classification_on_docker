# Docker Environment

## Build Docker

```
docker build -t mela_api:<tag> .
# docker build -t <image name> .   # build image 

# run docker without gpu
docker run \
-it \
--rm \
-p 12000:12000 \
-p 6006:6006 \
-v /home/linlin/dataset/sports_kaggle:/home/linlin/dataset/sports_kaggle \
-v /home/linlin/ll_docker/melanoma-deep-learning/docker_model/mela_api:/home/linlin/melanoma-deep-learning/mela_api \
mela_api:v2 

# docker run -v $host_path:$container_path
# -v /home/linlin/ll_docker/melanoma-deep-learning/docker_model/mela_api:/home/linlin/melanoma-deep-learning/mela_api \
mela_api:v1 -- mount the events files from mela_api into ./docker_model/mela_api folder (but files are still on docker container, not locally exist)
# -p host_port:container_port, 12000 for flask, 6006 for tensorboard
# run docker with gpu
```
## Train it 
```
python3 main.py
```


## Access the tensorboard in Docker  
- run `python3 -m tensorboard.main --logdir=. --bind_all` on container
- On the local pc: `localhost:6006`


## Run api 
```
python3 api.py 
```
- in local pc: go to `localhost:12000` to go to flask api 




## Acess 

## Modify the file inside the docker (so docker image would be updated) 

```
docker run -v /home/linlin/dataset/sports_kaggle/:/home/linlin/dataset/sports_kaggle/ \
-it \
--rm \
mela_api:latest  # docker run: create a new container 
```

- terminal 2 
```
docker exec -it mela_api:latest /bin/bash  # docker exec: run command on running container
# modify the file
# ctrl + d exit
docker commit -m "message" <container id> mela_api:latest 
```

## save the model weight from docker container to local 
```
# Syntax to Copy from Container to Docker Host  
docker cp {options} CONTAINER:SRC_PATH DEST_PATH 
# Syntax to Copy from Docker Host to Container  
docker cp {options} SRC_PATH CONTAINER:DEST_PATH 
```

## save docker image 
```
docker save mela_api:latest > docker_mela_api.tar
docker save myimage:latest | gzip > docker_image_mela_api.tar.gz
docker save mela_api:latest --output docker_image_mela_api.tar

docker load --input *.tar  #  It restores both images and tags.
docker load < *.tar
```