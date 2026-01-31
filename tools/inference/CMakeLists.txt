cmake_minimum_required(VERSION 3.18)

project(greta_inference_tools LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ROCm path
set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")

# Inference library location
set(INFERENCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../src/inference)

# Include directories
set(INFERENCE_INCLUDE_DIRS
    ${INFERENCE_DIR}/include
    ${INFERENCE_DIR}/../rt/backend/hip/include
    ${ROCM_PATH}/include
)

# Inference sources
set(INFERENCE_SOURCES
    ${INFERENCE_DIR}/src/weight_loader.cpp
    ${INFERENCE_DIR}/src/block_scheduler.cpp
    ${INFERENCE_DIR}/src/tokenizer.cpp
    ${INFERENCE_DIR}/src/generator.cpp
    ${INFERENCE_DIR}/../rt/backend/hip/src/buffer.cpp
)

# greta_infer CLI
add_executable(greta_infer
    src/greta_infer.cpp
    ${INFERENCE_SOURCES}
)
target_include_directories(greta_infer PRIVATE ${INFERENCE_INCLUDE_DIRS})
target_compile_definitions(greta_infer PRIVATE 
    GCORE_USE_HIP=1
    __HIP_PLATFORM_AMD__=1
)
target_link_directories(greta_infer PRIVATE ${ROCM_PATH}/lib)
target_link_libraries(greta_infer PRIVATE amdhip64)
