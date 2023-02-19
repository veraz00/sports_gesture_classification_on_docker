```
docker build -t mela_api:<tag> .
docker build -t mela_api .   # build image 
docker run -v /home/linlin/dataset/sports_kaggle/:/home/linlin/dataset/sports_kaggle/ \
-it \
--rm \
mela_api:latest
```

## Modify the file inside the docker (so docker image would be updated) 

```
docker run -v /home/linlin/dataset/sports_kaggle/:/home/linlin/data \
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

- specific character (refer: https://steemit.com/bash/@elliotyagami/bash-difference-between-or-and-or-or-and-and-and)

    - | : It is the pipe operator. It passes the stdout of first command to the next command : `docker save myimage:latest | gzip > docker_mela_api.tar.gz`
    - ||: It is like the boolean or operator. If the first half succeeds then don't executable the second half.
    - &&: It is like the boolean and operator. Executable both halves.
    - `&`: It starts a asynchronous process.





## use the current script to build docker image 