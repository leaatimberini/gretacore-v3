cmake_minimum_required(VERSION 3.18)

project(greta_inference_tools LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ROCm path
set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")

# Inference library location
set(INFERENCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../src/inference)
set(RT_HIP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../src/rt/backend/hip)

# Include directories
set(INFERENCE_INCLUDE_DIRS
    ${INFERENCE_DIR}/include
    ${RT_HIP_DIR}/include
    ${ROCM_PATH}/include
)

# Inference sources
set(INFERENCE_SOURCES
    ${INFERENCE_DIR}/src/weight_loader.cpp
    ${INFERENCE_DIR}/src/block_scheduler.cpp
    ${INFERENCE_DIR}/src/tokenizer.cpp
    ${INFERENCE_DIR}/src/generator.cpp
    ${RT_HIP_DIR}/src/buffer.cpp
)

# HIP kernel sources
set(HIP_KERNEL_SOURCES
    ${RT_HIP_DIR}/kernels/basic_kernels.hip
    ${RT_HIP_DIR}/kernels/gemm_kernels.hip
    ${RT_HIP_DIR}/kernels/attention_kernels.hip
)

# greta_infer CLI
add_executable(greta_infer
    src/greta_infer.cpp
    ${INFERENCE_SOURCES}
    ${HIP_KERNEL_SOURCES}
)
target_include_directories(greta_infer PRIVATE ${INFERENCE_INCLUDE_DIRS})
target_compile_definitions(greta_infer PRIVATE 
    GCORE_USE_HIP=1
    __HIP_PLATFORM_AMD__=1
)
target_link_directories(greta_infer PRIVATE ${ROCM_PATH}/lib)
target_link_libraries(greta_infer PRIVATE amdhip64)
