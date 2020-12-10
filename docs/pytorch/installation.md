# Install Pytorch in AWS from Source
[Official Instruction](https://github.com/pytorch/pytorch#from-source)
## Update OS
```shell script
sudo yum update
```
## Installation
### Python Environment
* Install python
```shell script
sudo yum install python3.7
```
* Install conda
```shell script
cd ~
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh
export PATH=~/anaconda3/bin:$PATH
```
* Install python modules
```shell script
conda install -c pytorch magma-cuda90 
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi
```
### Dev Libs
* Install official libs
```shell script
sudo yum groupinstall "Development Tools"
sudo yum install glog
sudo yum install protobuf-devel leveldb-devel snappy-devel opencv-devel boost-devel hdf5-devel
sudo yum install python3-devel
```
* Create path for source libs
```shell script
mkdir -p ~/workspaces/workspace_cpp  
```
#### Install libs from sources
* lmdb
```shell script
cd ~/workspaces/workspace_cpp/
git clone https://github.com/LMDB/lmdb.git
cd lmdb/libraries/liblmdb/
make
sudo make install
```
* gflags
```shell script
cd ~/workspaces/workspace_cpp/
git clone https://github.com/gflags/gflags.git
cd gflags/
mkdir build
cd build/
export CXXFLAGS="-fPIC" && cmake .. && make
sudo make install
```
* glog
```shell script
cd ~/workspaces/workspace_cpp/
clone https://github.com/google/glog
cd glog/
mkdir build && cd build
export CXXFLAGS="-fPIC" && cmake .. && make
sudo make install
```
* matplotlibcpp (optional)
```shell script
sudo pip3 install matplotlib
cd ~workspaces/workspace_cpp/
git clone https://github.com/lava/matplotlib-cpp.git
```
### Install pytorch
```shell script
cd ~/workspaces/workspace_cpp/
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git submodule sync
git submodule update --init --recursive
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
USE_LMDB=ON python3 setup.py install --cmake
```
## Set environment 
Append following command to *~/.bashrc*
```shell script
export LD_LIBRARY_PATH=/usr/local/lib
```
## Makefile
Add following settings in *CMakeLists.txt*
```shell script
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project (YOURPROJNAME)

set(WORKSPACE_PATH /home/ec2-user/workspaces/workspace_cpp)
set(PYTORCH_SRC_PATH ${WORKSPACE_PATH}/pytorch)
set(PYTORCH_SRC_INC_PATH ${PYTORCH_SRC_PATH}/torch/include/)

set(PYTORCH_BLD_PATH ${PYTORCH_SRC_PATH}/build/)
set(PYTORCH_GEN_INC_PATH ${PYTORCH_BLD_PATH}/lib.linux-x86_64-3.7/torch/include/)
set(PYTORCH_GEN_LIB_PATH ${PYTORCH_BLD_PATH}/lib/)
set(PYTORCH_LIB_PATH ${PYTORCH_BLD_PATH}/lib.linux-x86_64-3.7/torch/lib/)
set(STATIC_LIB_PATH ${PYTORCH_GEN_LIB_PATH})

set(static_libs ${STATIC_LIB_PATH}/libprotoc.a 
	${STATIC_LIB_PATH}/libprotobuf.a 
	${STATIC_LIB_PATH}/libpthreadpool.a
	${STATIC_LIB_PATH}/libc10d.a
	)

set(shared_libs opencv_core opencv_highgui opencv_imgproc torch torch_cpu glog gflags boost_system c10 rt pthread lmdb python3.7m stdc++fs)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-g -std=c++17")
set(CMAKE_BUILD_TYPE Debug)

include_directories(include)
include_directories(${CAFFE2_SRC_INC_PATH})
include_directories(${CAFFE2_GEN_INC_PATH})
include_directories(${CAFFE2_SRC_PATH}/torch/csrc/api/include)
include_directories(${WORKSPACE_PATH}/matplotlib-cpp)
include_directories(/usr/include/python3.7m)
include_directories(/usr/local/lib64/python3.7/site-packages/numpy/core/include)


link_directories(${CAFFE2_LIB_PATH})
```