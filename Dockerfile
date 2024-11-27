FROM nvcr.io/nvidia/pytorch:23.03-py3

# ENV Variables
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/include/"

COPY /gcap /gcap
WORKDIR /gcap

RUN pip3 install -r requirements.txt