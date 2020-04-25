FROM jupyter/scipy-notebook

MAINTAINER iwasakishuto <cabernet.rock@gmail.com>

USER root

# Install system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      wget && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages and kerasy
ENV NB_USER kerasy
ENV NB_UID 1001

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p /src && \
    chown $NB_USER /src

USER $NB_USER

RUN pip install --upgrade pip && \
    pip install \
      numpy>=1.15.1 \
      scipy>=1.4.1 \
      seaborn>=0.10.0 \
      cython>=0.28.5 && \
      git clone git://github.com/iwasakishuto/Kerasy.git /src && pip install -e /src[tests] && \
      pip install git+git://github.com/iwasakishuto/Kerasy.git

ENV PYTHONPATH='/src/:$PYTHONPATH'

WORKDIR /data

CMD jupyter notebook --port=8888 --ip=0.0.0.0
