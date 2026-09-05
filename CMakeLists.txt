cmake_minimum_required(VERSION 3.24)

# Default CUDA path from the course; change this to match your machine.
set(CMAKE_CUDA_COMPILER "/opt/cuda/bin/nvcc")
set(CUDACXX "/opt/cuda/bin/nvcc")

project(my-tiny-vllm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Ampere=80, Ada=89, Blackwell=120 — set to your GPU.
set(CMAKE_CUDA_ARCHITECTURES 120)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CUDA_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -DDEBUG")

find_package(CUDAToolkit REQUIRED)

add_executable(my-tiny-vllm
    src/main.cpp
    src/kernels.cu
)

target_include_directories(my-tiny-vllm PRIVATE src include)
target_link_libraries(my-tiny-vllm PRIVATE CUDA::cublas CUDA::cudart)
