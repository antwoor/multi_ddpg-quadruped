FROM tensorflow/tensorflow:latest-gpu 

# Update the package index and install some necessary packages

RUN apt-get update && \
    apt-get install -y \
        software-properties-common \
        build-essential \
        cmake \
        git \
        curl\
        wget


WORKDIR /tmp

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update 
RUN apt-get update && \
  apt-get install -y \
        cuda-drivers-550\
        nvidia-container-toolkit
RUN nvidia-ctk runtime configure --runtime=docker  




# Install Python 3.10
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y python3.10 python3-pip
# RUN pip3 install --upgrade pip   

# set WORKDIR
WORKDIR /home/root/rl_ws

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

# replace mirror
RUN apt-get install -q -y --no-install-recommends wget \
&& wget http://archive.ubuntu.com/ubuntu/pool/main/c/ca-certificates/ca-certificates_20230311ubuntu0.22.04.1_all.deb \
&& dpkg -i ./ca-certificates_20230311ubuntu0.22.04.1_all.deb
RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak && \
  echo "deb http://archive.ubuntu.com/ubuntu/ jammy main restricted universe multiverse\n" > /etc/apt/sources.list && \
  echo "deb http://archive.ubuntu.com/ubuntu/ jammy-updates main restricted universe multiverse\n" >> /etc/apt/sources.list && \
  echo "deb http://archive.ubuntu.com/ubuntu/ jammy-backports main restricted universe multiverse\n" >> /etc/apt/sources.list && \
  echo "deb http://archive.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse\n" >> /etc/apt/sources.list

# dependencies for gym
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
 libxcursor-dev \
 libxrandr-dev \
 libxinerama-dev \
 libxi-dev \
 mesa-common-dev \
 zip \
 unzip \
 make \
#  gcc-8 \
#  g++-8 \
#  vulkan-utils \
#  mesa-vulkan-drivers \
#  pigz \
 git \
 libegl1 \
 git-lfs \
 libsm6 \
 xauth \
 x11-apps

# pip install requirements
COPY requirement.txt .
RUN pip install -r requirement.txt


RUN ln -sfn /usr/bin/python3.10 /usr/bin/python

# RUN chmod +x setup.sh

ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
