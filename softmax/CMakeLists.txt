cmake_minimum_required(VERSION 3.24)
project(softmax LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)

# Collect all kernel source files
file(GLOB KERNEL_SOURCES "${CMAKE_SOURCE_DIR}/kernels/*.cu")

# Include CPM
include(cmake/CPM.cmake)

# Fetch nvbench
CPMAddPackage(
  NAME nvbench
  GITHUB_REPOSITORY NVIDIA/nvbench
  GIT_TAG main
  OPTIONS
    "NVBench_ENABLE_EXAMPLES OFF"
    "NVBench_ENABLE_TESTING OFF"
)

# Create executable
add_executable(softmax
    bench.cu
    ${KERNEL_SOURCES}
)

# Set target properties
set_target_properties(softmax PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)

# Target include directories
target_include_directories(softmax PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Target compile options for CUDA
target_compile_options(softmax PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

target_link_libraries(softmax PRIVATE nvbench::main)
