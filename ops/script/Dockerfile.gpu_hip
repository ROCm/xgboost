#
# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
# 
# Build instructions: https://confluence.amd.com/display/DCGPUAIST/XGBOOST+ROCm+Build
#
# Due to submodules of xgboost is currently in AMD-AI repository that cannot be directly cloned,
# we need to git clone the xgboost yourself before running docker build.
# Eventually if xgboost is in a public repository, you would be able to save this step.
# Please do the following to build this docker
#
# git clone --recursive git@github.com:AMD-AI/xgboost.git
# cd xgboost
# git checkout amd-condition
# git submodule update --init --recursive
# docker build --build-arg GITHUB_TOKEN=${GITHUB_TOKEN} -t mun-node-0.acp.amd.com:8001/xgboost:amd-condition -f Dockerfile .

FROM rocm/dev-ubuntu-20.04:5.4.2

#ENV GITHUB_TOKEN=<PLACEHOLDER_GET_FROM_BUILD_ARG>
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=/opt/rocm/rocm/lib:/opt/rocm/rocm/lib64:/opt/rocm/rocm/hip/lib:/opt/rocm/rocm/llvm/lib:/opt/rocm/rocm/opencl/lib:/opt/rocm/rocm/hcc/lib:/opt/rocm/rocm/opencl/lib/x86_64:${LD_LIBRARY_PATH}

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
	wget \
	git \
	ssh \
	cmake \
	vim \
        rocthrust \
        rocprim \
        hipcub \
	libgtest-dev \
	googletest \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
ENV VER1=3.26
ENV VER2=3.26.2
RUN wget -nv https://cmake.org/files/v${VER1}/cmake-${VER2}-linux-x86_64.tar.gz \
  && tar xf cmake-${VER2}-linux-x86_64.tar.gz \
  && ln -s cmake-${VER2}-linux-x86_64 cmake
ENV PATH="/opt/cmake/bin:${PATH}"

WORKDIR /opt/xgboost
COPY . .
ENV CMAKE_PREFIX_PATH=/opt/rocm/lib/cmake:/opt/rocm/lib/cmake/AMDDeviceLibs:${CMAKE_PREFIX_PATH}
#RUN git config --global user.name $USER
RUN git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"
RUN git config --global --unset url."https://${GITHUB_TOKEN}@github.com/".insteadOf
#RUN git clone https://${GITHUB_TOKEN}@github.com/AMD-AI/xgboost.git -b amd-condition --recurse-submodules \
# && cd xgboost \
RUN rm -fr build \
 && mkdir build \
 && cd build \
 && cmake .. -DUSE_HIP=ON -DGOOGLE_TEST=ON -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:/opt/rocm \
 && make -j
#ENV OMP_NUM_THREADS=8
#RUN build/testxgboost
WORKDIR /opt/xgboost/python-package/
RUN pip install -e .
