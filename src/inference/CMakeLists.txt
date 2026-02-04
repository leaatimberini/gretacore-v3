cmake_minimum_required(VERSION 3.18)

project(gcore_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ROCm path
set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")

# Include directories
set(INFERENCE_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../rt/backend/hip/include
    ${ROCM_PATH}/include
)

# Inference library sources
set(INFERENCE_SOURCES
    src/weight_loader.cpp
    src/block_scheduler.cpp
    src/tokenizer.cpp
    src/generator.cpp
    src/layer_trace.cpp
    src/stage_trace.cpp
)

# Build as static library
add_library(gcore_inference STATIC ${INFERENCE_SOURCES})
target_include_directories(gcore_inference PUBLIC ${INFERENCE_INCLUDE_DIRS})
target_link_directories(gcore_inference PUBLIC ${ROCM_PATH}/lib)
target_link_libraries(gcore_inference PRIVATE amdhip64)

# Weight Loader Test
add_executable(weight_loader_test
    test/weight_loader_test.cpp
    ${INFERENCE_SOURCES}
    ../../rt/backend/hip/src/buffer.cpp
)
target_include_directories(weight_loader_test PRIVATE ${INFERENCE_INCLUDE_DIRS})
target_compile_definitions(weight_loader_test PRIVATE 
    GCORE_USE_HIP=1
    __HIP_PLATFORM_AMD__=1
)
target_link_directories(weight_loader_test PRIVATE ${ROCM_PATH}/lib)
target_link_libraries(weight_loader_test PRIVATE amdhip64)

# Block Scheduler Test
add_executable(block_scheduler_test
    test/block_scheduler_test.cpp
    ${INFERENCE_SOURCES}
    ../../rt/backend/hip/src/buffer.cpp
)
target_include_directories(block_scheduler_test PRIVATE ${INFERENCE_INCLUDE_DIRS})
target_compile_definitions(block_scheduler_test PRIVATE 
    GCORE_USE_HIP=1
    __HIP_PLATFORM_AMD__=1
)
target_link_directories(block_scheduler_test PRIVATE ${ROCM_PATH}/lib)
target_link_libraries(block_scheduler_test PRIVATE amdhip64)

# Tokenizer Test (no HIP dependency)
add_executable(tokenizer_test
    test/tokenizer_test.cpp
    src/tokenizer.cpp
)
target_include_directories(tokenizer_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
