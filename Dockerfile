FROM nvcr.io/nvidia/deepstream-l4t:5.1-21.02-base
RUN mkdir /workdir
WORKDIR /workdir

# Building and installing latest CMake
RUN apt update -y && apt install wget git tar gcc g++ build-essential libssl-dev -y 
RUN wget https://github.com/Kitware/CMake/releases/download/v3.21.0-rc3/cmake-3.21.0-rc3.tar.gz
RUN tar -xvf cmake-3.21.0-rc3.tar.gz
RUN cd cmake-3.21.0-rc3 && ./bootstrap --parallel=$(nproc) && make && make install -j $(nproc)

# Installing tensorrt utils
RUN apt install pkg-config
ENV CC /usr/bin/gcc
ENV TRT_LIBPATH /usr/lib/aarch64-linux-gnu/ 
RUN pkg-config zlib

RUN git clone https://github.com/NVIDIA/TensorRT
RUN cd TensorRT && git checkout release/7.1 && git submodule update --init --recursive
RUN mkdir TensorRT/build 
RUN cd TensorRT/build && cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DTRT_PLATFORM_ID=aarch64 -DCUDA_VERSION=10.2 && make -j $(nproc) 
ENV PATH /usr/src/tensorrt/bin/:$PATH


# Installing python enviroment
RUN apt update && apt install python3-dev python3-pip python3-numpy python3-opencv -y
RUN pip3 install MarkupSafe==1.0
RUN pip3 install pycuda
ENV LD_PRELOAD /usr/lib/aarch64-linux-gnu/libgomp.so.1

COPY . /workdir
RUN python3 setup.py install
WORKDIR /workdir
