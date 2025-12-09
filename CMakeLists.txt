cmake_minimum_required(VERSION 3.10)
project(mytorch LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 89) # 4070

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(pybind11)
find_package(CUDAToolkit REQUIRED)
link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/x64")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -O3 -restrict")



set(SRC
    bind.cu
    src/tensor.cu
    src/nn.cu
)

pybind11_add_module(mytorch ${SRC})
target_compile_options(mytorch PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -use_fast_math
        -O3
        -restrict
    >
)


target_include_directories(mytorch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/dlpack/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# === 链接 CUDA 库 ===
target_link_libraries(mytorch PRIVATE
    cublas
    curand
)


message(STATUS "✅ Building mytorch with pybind11_add_module + CUDA")
message(STATUS "🔧 CUDA Arch: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "🔧 CUDA Flags: ${CMAKE_CUDA_FLAGS}")