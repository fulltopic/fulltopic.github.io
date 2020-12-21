cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project (ch6)

set(CAFFE2_SRC_PATH /home/zf/workspaces/workspace_cpp/pytorch_1_8/pytorch)
set(CAFFE2_SRC_INC_PATH ${CAFFE2_SRC_PATH}/torch/include/)

set(CAFFE2_BLD_PATH ${CAFFE2_SRC_PATH}/build/)
set(CAFFE2_GEN_INC_PATH ${CAFFE2_BLD_PATH}/lib.linux-x86_64-3.8/torch/include/)
set(CAFFE2_GEN_LIB_PATH ${CAFFE2_BLD_PATH}/lib/)
set(CAFFE2_LIB_PATH ${CAFFE2_BLD_PATH}/lib.linux-x86_64-3.8/torch/lib/)
set(STATIC_LIB_PATH ${CAFFE2_GEN_LIB_PATH})

set(static_libs ${STATIC_LIB_PATH}/libprotoc.a 
	${STATIC_LIB_PATH}/libprotobuf.a 
	${STATIC_LIB_PATH}/libpthreadpool.a
	${STATIC_LIB_PATH}/libc10d.a)
set(shared_libs opencv_core opencv_highgui opencv_imgcodecs opencv_imgproc torch torch_cpu glog gflags boost_system  boost_filesystem boost_thread c10 rt pthread leveldb lmdb python3.9)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-g -std=c++17")
set(CMAKE_BUILD_TYPE Debug)

#set(PYTORCH_CPP_RL_PATH /home/zf/workspaces/workspace_cpp/pytorch-cpp-rl)
#set(PYTORCH_CPP_RL_LIB_PATH ${PYTORCH_CPP_RL_PATH}/example/lib)
#set(pytorch_cpp_rl_lib ${PYTORCH_CPP_RL_PATH}/build/libcpprl.a ${PYTORCH_CPP_RL_PATH}/build/example/lib/libzmq/lib/libzmq.a)


include_directories(./)
include_directories(${CAFFE2_SRC_INC_PATH})
include_directories(${CAFFE2_GEN_INC_PATH})
include_directories(${CAFFE2_SRC_PATH}/torch/csrc/api/include)
include_directories(/home/zf/workspaces/workspace_cpp/matplotlib-cpp)
include_directories(/usr/include/python3.9)
include_directories(${PYTORCH_CPP_RL_PATH}/include)
include_directories(${PYTORCH_CPP_RL_PATH}/example)
include_directories(${PYTORCH_CPP_RL_LIB_PATH}/msgpack-c/include)

link_directories(${CAFFE2_LIB_PATH})

add_executable(windgrid ./windgrid.cpp)
target_compile_features(windgrid PUBLIC cxx_range_for)
target_link_libraries(windgrid ${shared_libs})
