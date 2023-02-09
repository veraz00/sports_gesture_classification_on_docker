# Steps on Building Docker Image 
- every time when changing code locally, re-run docker build to update docker image 

- how to acess the dataset 
    - volumes are folder on the host machine which can be mapped into docker container 

- Docker image is named with <image name>:<tag name>, if no tag name, tag name = latest 
- Container is from docker image <image name>:<tag name>, then it would be added with a container name 


```
docker build -f Dockerfile -t linlin/mela_train:version1 .    
# build a docker image named as mela_train
# docker build -t <USER>/<CONTAINER>:<VERSION>  


docker run --gpus all -v /home/linlin/dataset/sports_kaggle/:/home/linlin/data -ti linlin/mela_train:version1 python3 main.py nvidia-smi --rm
# -it is short for --interactive + --tty . When you docker run with this command it takes you straight inside the container.
# -d is short for --detach , which means you just run the container and then detach from it. Essentially, you run container in the background
# --rm	Automatically remove the container when it exits
# docker run image_name:tag_name. If you didn't specify tag_name it will automatically run an image with the 'latest' tag. Instead of image_name , you can also specify an image ID (no tag_name).

```

## command 
```
# check the docker images  vs docker images ls
docker images -a 

# remove docker images 
docker rmi <docker id: 58db3edaf>
docker rmi $(docker images -q)  # -q: list only image id

# check the docker containers 
docker ps -a 
docker stop <container id>
docker stop $(docker ps -a)
docker rm <container_id>
```



## How create virtual environment in docker 
```
in Dockerfile
RUN cd /home/linlin/app/ && source mela_env/bin/activate
# do not copy environment into docker file, else: create a virtual environment and pip install and activate it 

FROM python:3.9-slim-bullseye

RUN python3 -m venv /opt/venv

# # Install dependencies:
COPY requirements.txt .
RUN . /opt/venv/bin/activate && pip install -r requirements.txt

# # Run the application:
COPY myapp.py .
CMD . /opt/venv/bin/activate && exec python myapp.py
```