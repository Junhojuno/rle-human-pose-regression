# base image
FROM tensorflow/tensorflow:2.11.0

# 기본 패키지 설치
RUN apt-get update && \
    apt-get install -y git && \
    pip install -U git+https://github.com/albumentations-team/albumentations

RUN pip install tensorflow-addons==0.19.0 && \
    pip install pycocotools && \
    python -m pip install opencv-python && \
    pip install --upgrade wandb && \
    pip install tqdm && \
    pip install easydict

# tcmalloc
RUN apt-get install libtcmalloc-minimal4
