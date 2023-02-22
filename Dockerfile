# image source: https://hub.docker.com/r/nvidia/cuda
FROM nvidia/cuda:11.6.1-base-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    sudo \
    build-essential \
    wget \
    vim

# The -m option of useradd command allows to copy all files from your system skeleton directory (/etc/skel) to the newly created home directory.
RUN useradd -m linlin 

# chown: add another layer on image 
# chown -R <past_owner>:<current_owner> the target directory
RUN chown -R linlin:linlin /home/linlin/  

# do not copy environment into docker file, instead, create a virtual environment and pip install and activate it 
COPY --chown=linlin . /home/linlin/melanoma-deep-learning/

USER linlin

RUN cd /home/linlin/melanoma-deep-learning/ && pip3 install -r requirements.txt

WORKDIR /home/linlin/melanoma-deep-learning


