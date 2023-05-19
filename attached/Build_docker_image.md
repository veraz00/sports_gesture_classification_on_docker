# Run Sport Gesture Classification on Docker
- I did not install `nvidia runtime` to let docker access the local gpu. Now it would only use cpu device.

## Install Docker Server & Client 
```
# remove the installed docker
sudo apt-get purge docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin docker-ce-rootless-extras
sudo rm -rf /var/lib/docker
sudo rm -rf /var/lib/containerd

# install
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# check docker version
docker version
```
![](docker_version.jpg)


- If meeting any errors, you can check [Installation.md](attatched/Installation.md)

## Build Docker

```
# build image 
docker build -t sports_api:<tag> .

# run docker without gpu
docker run \
-it \
--rm \
-p 12000:12000 \
-p 6006:6006 \
-v /home/linlin/dataset/sports_kaggle:/home/linlin/dataset/sports_kaggle \
sports_api:v1
# -t gives the sudo terminal
# -i: give the interacting interface
# -p host_port:container_port, 12000 for flask, 6006 for tensorboard
# -v $host_path:$container_path
```

## Train in Docker Container 
```
python3 train.py
```


## Tensorboard Events
- Here the local logging events would be saved into docker with the same directory as `/home/linlin/ll_docker/sportsnoma-deep-learning/sports_events` 
- The new events file when training on docker would be saved on this folder in docker, but files are still on docker container, not locally exist


### Access the Tensorboard in Docker  
- Run `python3 -m tensorboard.main --logdir=. --bind_all` on container
- In local pc: `localhost:6006`



## Data Transfer between container and local
```
sudo docker cp container-id:/path/filename.txt ~/Desktop/filename.txt
sudo docker cp foo.txt container_id:/foo.txt
```

## Run API in Docker Container
```
python3 api.py 
```
- In local pc: go to `localhost:12000` to go to flask api 


## Acess 

- Modify the file inside the docker (so docker image would be updated) 

    - terminal 1
    ```
    docker run -v /home/linlin/dataset/sports_kaggle/:/home/linlin/dataset/sports_kaggle/ \
    -it \
    --rm \
    sports_api:<tag>  # docker run: create a new container    
    ```

    - terminal 2 
    ```
    docker exec -it <container_id> /bin/bash  # docker exec: run command on running container
    # modify the file
    # ctrl + d exit
    docker commit -m "message" <container id> sports_api:<tag>
    ```


## Save Image
```
docker save sports_api:<tag> > docker_sports_api.tar
docker save myimage:<tag> | gzip > docker_image_sports_api.tar.gz
docker save sports_api:<tag> --output docker_image_sports_api.tar

docker load --input *.tar  #  It restores both images and tags.
docker load < *.tar
```