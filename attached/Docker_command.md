# Docker 
## Images 
```
# list images 
docker images 

# delete docker images 
docker rmi <IMAGE ID> 
python3 -m tensorboard.main --logdir=. --bind_all
# build image 
docker build -t sports_api:<tag> .
docker run python-fastapi 
```
error: use the docker to run; it shows the web page cannot be opened 
- Solution: `docker run -p 8000:8000 python-fastapi`


## Containers
```
docker ps # get the list of container 

# The exec command is used to interact with already running containers on the Docker host
sudo docker exec -it <container-id> /bin/sh  # go into the container inside
# -t gives the sudo terminal
# -i: give the interacting interface

docker stop <container_id>

```

## Data Transfer between container and local
```
sudo docker cp container-id:/path/filename.txt ~/Desktop/filename.txt
sudo docker cp foo.txt container_id:/foo.txt
```

## Pip
```
pip freeze > requirements.txt
```
pip freeze vs pip list 
the two packages shown in pip list but not pip freeze are setuptools (which is easy_install) and pip itself




