FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

LABEL author="Marcelo Garcia"

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update --fix-missing && apt-get install -y \
    ffmpeg \
    git \
    git-core \ 
    pkg-config \
    python3-pip \
    unzip \
    vim \
    wget \
    zip \
    zlib1g-dev \
    libcurl4 \
    curl \
    ffmpeg

WORKDIR /project

COPY requirements.txt .

RUN pip3 install -r requirements.txt

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
