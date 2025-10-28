cmake_minimum_required(VERSION 3.10)
project(mytorch LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 89) # 4070

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(pybind11)
find_package(CUDAToolkit REQUIRED)
link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/x64")



set(SRC
    bind.cu
    src/tensor.cu
)

pybind11_add_module(mytorch ${SRC})

set_target_properties(mytorch PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(mytorch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

# === 链接 CUDA 库 ===
target_link_libraries(mytorch PRIVATE
    cublas
    curand
)

message(STATUS "✅ Building mytorch with pybind11_add_module + CUDA")
