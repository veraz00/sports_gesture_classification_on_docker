# image source: https://hub.docker.com/r/nvidia/cuda
FROM nvidia/cuda:12.0.1-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    sudo \
    build-essential \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    autoconf \
    git \
    wget

# The -m option of useradd command allows to copy all files from your system skeleton directory (/etc/skel) to the newly created home directory.
RUN useradd -m linlin 

RUN chown -R linlin:linlin /home/linlin/

COPY --chown=linlin . /home/linlin/app/

USER linlin

RUN cd /home/linlin/app/ && pip3 install -r requirements.txt

RUN git clone --branch 22.06-dev https://github.com/NVIDIA/apex.git
RUN cd apex
RUN python3 setup.py install
RUN cd /home/linlin/app/

WORKDIR /home/linlin/app


