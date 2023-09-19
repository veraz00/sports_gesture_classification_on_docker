# image source: https://hub.docker.com/r/nvidia/cuda
# https://hub.docker.com/layers/nvidia/cuda/11.0.3-cudnn8-runtime-rockylinux8/images/sha256-edba79371ee1d668fd1c14b12dd6792899c14bb4fcc979edc233d76eed414701?context=explore

FROM ubuntu:latest

RUN apt-get update && apt-get install -y git && \
    apt-get install -y python3-pip sudo vim wget && \ 
    rm -rf /var/lib/apt/lists/*

# The -m option of useradd command allows to copy all files from your system skeleton directory (/etc/skel) to the newly created home directory.
RUN useradd -m linlin 

# chown: add another layer on image 
# chown -R <past_owner>:<current_owner> the target directory
RUN chown -R linlin:linlin /home/linlin/  


# COPY --chown=<user>:<group> <hostPath> <containerPath>    
# do not copy environment into docker file, add environment path into `.dockerignore`
COPY --chown=linlin . /home/linlin/sport_gesture_classification_on_docker/

USER linlin

RUN cd /home/linlin/sport_gesture_classification_on_docker/ && pip3 install -r requirements.txt

WORKDIR /home/linlin/sport_gesture_classification_on_docker/
