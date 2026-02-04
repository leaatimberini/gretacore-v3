cmake_minimum_required(VERSION 3.18)

project(greta_inference_tools LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

# Force MI300X architecture
set(CMAKE_HIP_ARCHITECTURES "gfx942")

# ROCm path
set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")

# Inference library location
set(INFERENCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../src/inference)
set(RT_HIP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../src/rt/backend/hip)

# Include directories
set(INFERENCE_INCLUDE_DIRS
    ${INFERENCE_DIR}/include
    ${RT_HIP_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/rt/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/compute/include
    ${ROCM_PATH}/include
)

# Inference sources
set(INFERENCE_SOURCES
    ${INFERENCE_DIR}/src/weight_loader.cpp
    ${INFERENCE_DIR}/src/block_scheduler.cpp
    ${INFERENCE_DIR}/src/tokenizer.cpp
    ${INFERENCE_DIR}/src/generator.cpp
    ${INFERENCE_DIR}/src/layer_trace.cpp
    ${INFERENCE_DIR}/src/stage_trace.cpp
    ${RT_HIP_DIR}/src/buffer.cpp
    ${RT_HIP_DIR}/src/greta_runtime_hip.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/compute/src/greta_compute_hip.cpp
)

# HIP kernel sources
set(HIP_KERNEL_SOURCES
    ${RT_HIP_DIR}/kernels/basic_kernels.hip
    ${RT_HIP_DIR}/kernels/gemm_kernels.hip
    ${RT_HIP_DIR}/kernels/attention_kernels.hip
    ${RT_HIP_DIR}/kernels/fused_compute_kernels.hip
    ${RT_HIP_DIR}/kernels/fused_attention_kernels.hip
)

# Optional SentencePiece tokenizer
option(GRETA_USE_SENTENCEPIECE "Enable SentencePiece tokenizer" ON)

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
target_link_libraries(greta_infer PRIVATE amdhip64 OpenMP::OpenMP_CXX)

# SentencePiece linkage
if(GRETA_USE_SENTENCEPIECE)
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(SENTENCEPIECE sentencepiece)
        if(SENTENCEPIECE_FOUND)
            target_compile_definitions(greta_infer PRIVATE GRETA_USE_SENTENCEPIECE)
            target_include_directories(greta_infer PRIVATE ${SENTENCEPIECE_INCLUDE_DIRS})
            target_link_libraries(greta_infer PRIVATE ${SENTENCEPIECE_LIBRARIES})
            message(STATUS "SentencePiece enabled: ${SENTENCEPIECE_LIBRARIES}")
        else()
            message(WARNING "SentencePiece not found, using ASCII fallback")
        endif()
    endif()
endif()

# test_l0
add_executable(test_l0
    test_l0.cpp
    ${RT_HIP_DIR}/src/greta_runtime_hip.cpp
)
target_include_directories(test_l0 PRIVATE ${INFERENCE_INCLUDE_DIRS})
target_compile_definitions(test_l0 PRIVATE 
    GCORE_USE_HIP=1
    __HIP_PLATFORM_AMD__=1
)
target_link_directories(test_l0 PRIVATE ${ROCM_PATH}/lib)
target_link_libraries(test_l0 PRIVATE amdhip64)
#TEST
