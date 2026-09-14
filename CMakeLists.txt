cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(my-tiny-vllm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

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

include(CTest)
if(BUILD_TESTING)
    add_executable(decode-softmax-test tests/decode_softmax.cu src/kernels.cu)
    target_include_directories(decode-softmax-test PRIVATE src)
    target_link_libraries(decode-softmax-test PRIVATE CUDA::cudart)
    add_test(NAME decode-softmax COMMAND decode-softmax-test)
endif()
