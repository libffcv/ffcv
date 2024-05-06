FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        curl \
        git \
        ffmpeg

RUN conda update conda && \
        conda create -n ffcv python=3.9 \
        cupy \
        pkg-config \
        compilers \
        libjpeg-turbo \
        opencv \
        pytorch \
        torchvision \
        cudatoolkit=11.3 \
        numba -c pytorch -c conda-forge 

RUN echo "source activate" >> ~/.bashrc
RUN echo "conda activate ffcv" >> ~/.bashrc

RUN git clone https://github.com/libffcv/ffcv.git  

RUN conda run -n ffcv pip install ffcv

# To test:
# 1- build the Dockerfile (e.g. docker build -t ffcv .)
# 2- login to the docker container (e.g. docker run -it --gpus all ffcv bash)
# 3- cd ffcv/examples/cifar
# 4- bash train_cifar.sh
