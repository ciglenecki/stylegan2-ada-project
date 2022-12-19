# we want to install as little as possible so that people without docker can use Python via requirements.txt
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
## Set timezone as it is required by some packages.
ENV TZ=Europe/Berlin
## CUDA architectures, required by tiny-cuda-nn.
ENV TCNN_CUDA_ARCHITECTURES=61
## CUDA Home, required to find CUDA in some packages.
ENV CUDA_HOME="/usr/local/cuda"

ARG USER_NAME
ARG USER_ID
ARG GROUP_NAME
ARG GROUP_ID

# Update and upgrade all packages
RUN apt update -y
RUN apt upgrade -y

# Install python
RUN apt install -y git python3 software-properties-common python3-pip python-is-python3

# Install other useful
RUN apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    htop \
    wget \
    build-essential \
    cmake

# Dependency on face-alignment
RUN apt-get install -y ffmpeg libsm6 libxext6

COPY requirements.txt /tmp/requirements.txt
COPY requirements-dev.txt /tmp/requirements-dev.txt

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements-dev.txt
# Fixes plugin not found stylegan2-ada issue
RUN apt-get intsall -y ninja-build

RUN apt-get install -y sudo
RUN addgroup --gid $GROUP_ID $GROUP_NAME
# Create a new user with name USER_NAME
RUN adduser --disabled-password --uid $USER_ID --gid $GROUP_ID --shell /bin/bash $USER_NAME

# Set password for a user
# RUN echo "$USER_NAME:PASSWORD" | chpasswd 

# Add a user as a new sudoer
RUN adduser $USER_NAME sudo

# Remove password
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Set the pretty and obvious prompts
ENV TERM xterm-256color
RUN echo 'export PS1="\A \[\033[1;36m\]\h\[\033[1;34m\] \w \[\033[0;015m\]\\$ \[$(tput sgr0)\]\[\033[0m\]"' >> "/home/$USER_NAME/.bashrc"

# Set user
USER $USER_NAME
# Set CWDIR
WORKDIR "/home/$USER_NAME"
CMD ["bash", "-l"]