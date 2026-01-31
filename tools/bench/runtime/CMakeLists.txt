cmake_minimum_required(VERSION 3.22)
project(greta_runtime_bench LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------------------------------------------------------
# Common includes (runtime core)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../../../src/rt/allocator/include)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../../../src/rt/stream/include)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../../../src/rt/telemetry/include)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../../../src/rt/dispatch/include)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../../../src/rt/backend/vulkan/include)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../../../src/rt/backend/hip/include)

# -------------------------------------------------------------------
# Core benches (non-Vulkan)
add_executable(alloc_bench
  src/alloc_bench.cpp
  ../../../src/rt/allocator/src/allocator.cpp
)
target_compile_options(alloc_bench PRIVATE -O3 -march=native -pthread)

add_executable(stream_bench
  src/stream_bench.cpp
  ../../../src/rt/stream/src/stream.cpp
)
target_compile_options(stream_bench PRIVATE -O3 -march=native -pthread)

add_executable(telemetry_bench
  src/telemetry_bench.cpp
  ../../../src/rt/telemetry/src/telemetry.cpp
)
target_compile_options(telemetry_bench PRIVATE -O3 -march=native -pthread)

add_executable(dispatch_bench
  src/dispatch_bench.cpp
  ../../../src/rt/dispatch/src/dispatch.cpp
  ../../../src/rt/stream/src/stream.cpp
  ../../../src/rt/telemetry/src/telemetry.cpp
)
target_compile_options(dispatch_bench PRIVATE -O3 -march=native -pthread)

add_executable(llm_primitives_bench
  src/llm_primitives_bench.cpp
)
target_compile_options(llm_primitives_bench PRIVATE -O3 -march=native -pthread)

# -------------------------------------------------------------------
# Vulkan
find_package(Vulkan REQUIRED)

add_executable(vk_smoke_bench
  src/vk_smoke_bench.cpp
  ../../../src/rt/backend/vulkan/src/backend.cpp
)
target_compile_options(vk_smoke_bench PRIVATE -O3 -march=native -pthread)
target_link_libraries(vk_smoke_bench PRIVATE Vulkan::Vulkan)

# -------------------------------------------------------------------
# HIP / ROCm
if(NOT DEFINED ROCM_PATH)
  if(NOT DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "/opt/rocm")
  else()
    set(ROCM_PATH $ENV{ROCM_PATH})
  endif()
endif()

# We use hipcc as the compiler for .cpp files that need HIP headers,
# or we use HIP as a package. Since we're likely building with a standard
# host compiler (g++) and calling hipcc/clang internally, 
# for benches we can just link to HIP.
find_package(HIP QUIET)

if(HIP_FOUND OR EXISTS ${ROCM_PATH})
  message(STATUS "HIP found at: ${ROCM_PATH}")
  include_directories(${ROCM_PATH}/include)
  link_directories(${ROCM_PATH}/lib)

  add_executable(hip_smoke_bench
    src/hip_smoke_bench.cpp
    ../../../src/rt/backend/hip/src/backend.cpp
    ../../../src/rt/backend/hip/src/buffer.cpp
    ../../../src/rt/backend/hip/src/stream.cpp
  )
  target_compile_definitions(hip_smoke_bench PRIVATE 
    GCORE_USE_HIP=1
    __HIP_PLATFORM_AMD__=1
  )
  target_compile_options(hip_smoke_bench PRIVATE -O3 -march=native -pthread)
  
  add_executable(hip_fill_bench
    src/hip_fill_bench.cpp
    ../../../src/rt/backend/hip/kernels/basic_kernels.hip
    ../../../src/rt/backend/hip/src/backend.cpp
    ../../../src/rt/backend/hip/src/buffer.cpp
    ../../../src/rt/backend/hip/src/stream.cpp
  )
  target_compile_definitions(hip_fill_bench PRIVATE 
    GCORE_USE_HIP=1
    __HIP_PLATFORM_AMD__=1
  )
  target_compile_options(hip_fill_bench PRIVATE -O3 -march=native -pthread)

  add_executable(hip_rmsnorm_bench
    src/hip_rmsnorm_bench.cpp
    ../../../src/rt/backend/hip/kernels/basic_kernels.hip
    ../../../src/rt/backend/hip/src/backend.cpp
    ../../../src/rt/backend/hip/src/buffer.cpp
    ../../../src/rt/backend/hip/src/stream.cpp
  )
  target_compile_definitions(hip_rmsnorm_bench PRIVATE 
    GCORE_USE_HIP=1
    __HIP_PLATFORM_AMD__=1
  )
  target_compile_options(hip_rmsnorm_bench PRIVATE -O3 -march=native -pthread)

  add_executable(hip_gemm_bench
    src/hip_gemm_bench.cpp
    ../../../src/rt/backend/hip/kernels/gemm_kernels.hip
    ../../../src/rt/backend/hip/src/backend.cpp
    ../../../src/rt/backend/hip/src/buffer.cpp
    ../../../src/rt/backend/hip/src/stream.cpp
  )
  target_compile_definitions(hip_gemm_bench PRIVATE 
    GCORE_USE_HIP=1
    __HIP_PLATFORM_AMD__=1
  )
  target_compile_options(hip_gemm_bench PRIVATE -O3 -march=native -pthread)

  add_executable(hip_graph_runner_test
    src/hip_graph_runner_test.cpp
    ../../../src/rt/backend/hip/kernels/basic_kernels.hip
    ../../../src/rt/backend/hip/kernels/gemm_kernels.hip
    ../../../src/rt/backend/hip/kernels/attention_kernels.hip
    ../../../src/rt/backend/hip/src/backend.cpp
    ../../../src/rt/backend/hip/src/buffer.cpp
    ../../../src/rt/backend/hip/src/stream.cpp
  )
  target_compile_definitions(hip_graph_runner_test PRIVATE 
    GCORE_USE_HIP=1
    __HIP_PLATFORM_AMD__=1
  )
  target_compile_options(hip_graph_runner_test PRIVATE -O3 -march=native -pthread)
  target_include_directories(hip_graph_runner_test PRIVATE 
    ${CMAKE_CURRENT_LIST_DIR}/../../../src/rt/graph/include
  )

  add_executable(hip_attention_bench
    src/hip_attention_bench.cpp
    ../../../src/rt/backend/hip/kernels/basic_kernels.hip
    ../../../src/rt/backend/hip/kernels/gemm_kernels.hip
    ../../../src/rt/backend/hip/kernels/attention_kernels.hip
    ../../../src/rt/backend/hip/src/backend.cpp
    ../../../src/rt/backend/hip/src/buffer.cpp
    ../../../src/rt/backend/hip/src/stream.cpp
  )
  target_compile_definitions(hip_attention_bench PRIVATE 
    GCORE_USE_HIP=1
    __HIP_PLATFORM_AMD__=1
  )
  target_compile_options(hip_attention_bench PRIVATE -O3 -march=native -pthread)
  target_include_directories(hip_attention_bench PRIVATE 
    ${CMAKE_CURRENT_LIST_DIR}/../../../src/rt/graph/include
  )

  add_executable(hip_kv_cache_bench
    src/hip_kv_cache_bench.cpp
    ../../../src/rt/backend/hip/kernels/basic_kernels.hip
    ../../../src/rt/backend/hip/kernels/gemm_kernels.hip
    ../../../src/rt/backend/hip/kernels/attention_kernels.hip
    ../../../src/rt/backend/hip/src/backend.cpp
    ../../../src/rt/backend/hip/src/buffer.cpp
    ../../../src/rt/backend/hip/src/stream.cpp
  )
  target_compile_definitions(hip_kv_cache_bench PRIVATE 
    GCORE_USE_HIP=1
    __HIP_PLATFORM_AMD__=1
  )
  target_compile_options(hip_kv_cache_bench PRIVATE -O3 -march=native -pthread)
  target_include_directories(hip_kv_cache_bench PRIVATE 
    ${CMAKE_CURRENT_LIST_DIR}/../../../src/rt/graph/include
  )

  add_executable(hip_llama_block_test
    src/hip_llama_block_test.cpp
    ../../../src/rt/backend/hip/kernels/basic_kernels.hip
    ../../../src/rt/backend/hip/kernels/gemm_kernels.hip
    ../../../src/rt/backend/hip/kernels/attention_kernels.hip
    ../../../src/rt/backend/hip/src/backend.cpp
    ../../../src/rt/backend/hip/src/buffer.cpp
    ../../../src/rt/backend/hip/src/stream.cpp
  )
  target_compile_definitions(hip_llama_block_test PRIVATE 
    GCORE_USE_HIP=1
    __HIP_PLATFORM_AMD__=1
  )
  target_compile_options(hip_llama_block_test PRIVATE -O3 -march=native -pthread)
  target_include_directories(hip_llama_block_test PRIVATE 
    ${CMAKE_CURRENT_LIST_DIR}/../../../src/rt/graph/include
  )

  # Try various ways to link HIP
  if(TARGET hip::host)
    target_link_libraries(hip_smoke_bench PRIVATE hip::host)
    target_link_libraries(hip_fill_bench PRIVATE hip::host)
  else()
    # Fallback to manual search
    find_library(HIP_LIB NAMES amdhip64 hip_hcc PATHS ${ROCM_PATH}/lib NO_DEFAULT_PATH)
    if(HIP_LIB)
      target_link_libraries(hip_smoke_bench PRIVATE ${HIP_LIB})
      target_link_libraries(hip_fill_bench PRIVATE ${HIP_LIB})
    else()
      # Final attempt: just try the name
      target_link_libraries(hip_smoke_bench PRIVATE amdhip64)
      target_link_libraries(hip_fill_bench PRIVATE amdhip64)
    endif()
  endif()
else()
  message(WARNING "HIP not found. hip_smoke_bench will not be built.")
endif()

# -------------------------------------------------------------------
# Shaders (glslangValidator -> SPIR-V)
find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)
set(SHADERS_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../src/rt/backend/vulkan/shaders)

set(FILL_GLSL ${SHADERS_DIR}/fill.comp.glsl)
set(FILL_SPV  ${CMAKE_CURRENT_BINARY_DIR}/fill.comp.spv)
add_custom_command(OUTPUT ${FILL_SPV}
  COMMAND ${GLSLANG_VALIDATOR} -V ${FILL_GLSL} -o ${FILL_SPV}
  DEPENDS ${FILL_GLSL}
  VERBATIM)

set(GEMM_GLSL ${SHADERS_DIR}/gemm_f32.comp.glsl)
set(GEMM_SPV  ${CMAKE_CURRENT_BINARY_DIR}/gemm_f32.comp.spv)
add_custom_command(OUTPUT ${GEMM_SPV}
  COMMAND ${GLSLANG_VALIDATOR} -V ${GEMM_GLSL} -o ${GEMM_SPV}
  DEPENDS ${GEMM_GLSL}
  VERBATIM)

set(GEMM_TILED_GLSL ${SHADERS_DIR}/gemm_f32_tiled.comp.glsl)
set(GEMM_TILED_SPV  ${CMAKE_CURRENT_BINARY_DIR}/gemm_f32_tiled.comp.spv)
add_custom_command(OUTPUT ${GEMM_TILED_SPV}
  COMMAND ${GLSLANG_VALIDATOR} -V ${GEMM_TILED_GLSL} -o ${GEMM_TILED_SPV}
  DEPENDS ${GEMM_TILED_GLSL}
  VERBATIM)

set(GEMM_F16ACC32_GLSL ${SHADERS_DIR}/gemm_f16acc32_tiled.comp.glsl)
set(GEMM_F16ACC32_SPV  ${CMAKE_CURRENT_BINARY_DIR}/gemm_f16acc32_tiled.comp.spv)
add_custom_command(OUTPUT ${GEMM_F16ACC32_SPV}
  COMMAND ${GLSLANG_VALIDATOR} -V ${GEMM_F16ACC32_GLSL} -o ${GEMM_F16ACC32_SPV}
  DEPENDS ${GEMM_F16ACC32_GLSL}
  VERBATIM)

set(GEMM_F16ACC32_VEC2_GLSL ${SHADERS_DIR}/gemm_f16acc32_tiled_vec2.comp.glsl)
set(GEMM_F16ACC32_VEC2_SPV  ${CMAKE_CURRENT_BINARY_DIR}/gemm_f16acc32_tiled_vec2.comp.spv)
add_custom_command(OUTPUT ${GEMM_F16ACC32_VEC2_SPV}
  COMMAND ${GLSLANG_VALIDATOR} -V ${GEMM_F16ACC32_VEC2_GLSL} -o ${GEMM_F16ACC32_VEC2_SPV}
  DEPENDS ${GEMM_F16ACC32_VEC2_GLSL}
  VERBATIM)

set(GEMM_F16ACC32_VEC2_32X8_GLSL ${SHADERS_DIR}/gemm_f16acc32_tiled_vec2_32x8.comp.glsl)
set(GEMM_F16ACC32_VEC2_32X8_SPV  ${CMAKE_CURRENT_BINARY_DIR}/gemm_f16acc32_tiled_vec2_32x8.comp.spv)
add_custom_command(OUTPUT ${GEMM_F16ACC32_VEC2_32X8_SPV}
  COMMAND ${GLSLANG_VALIDATOR} -V ${GEMM_F16ACC32_VEC2_32X8_GLSL} -o ${GEMM_F16ACC32_VEC2_32X8_SPV}
  DEPENDS ${GEMM_F16ACC32_VEC2_32X8_GLSL}
  VERBATIM)

set(GEMM_F16ACC32_VEC2_DB_GLSL ${SHADERS_DIR}/gemm_f16acc32_tiled_vec2_db.comp.glsl)
set(GEMM_F16ACC32_VEC2_DB_SPV  ${CMAKE_CURRENT_BINARY_DIR}/gemm_f16acc32_tiled_vec2_db.comp.spv)
add_custom_command(OUTPUT ${GEMM_F16ACC32_VEC2_DB_SPV}
  COMMAND ${GLSLANG_VALIDATOR} -V ${GEMM_F16ACC32_VEC2_DB_GLSL} -o ${GEMM_F16ACC32_VEC2_DB_SPV}
  DEPENDS ${GEMM_F16ACC32_VEC2_DB_GLSL}
  VERBATIM)

set(GEMM_F16ACC32_SUBGROUP_GLSL ${SHADERS_DIR}/gemm_f16acc32_subgroup.comp.glsl)
set(GEMM_F16ACC32_SUBGROUP_SPV  ${CMAKE_CURRENT_BINARY_DIR}/gemm_f16acc32_subgroup.comp.spv)
add_custom_command(OUTPUT ${GEMM_F16ACC32_SUBGROUP_SPV}
  COMMAND ${GLSLANG_VALIDATOR}
          --target-env vulkan1.1
          -V ${GEMM_F16ACC32_SUBGROUP_GLSL}
          -o ${GEMM_F16ACC32_SUBGROUP_SPV}
  DEPENDS ${GEMM_F16ACC32_SUBGROUP_GLSL}
  VERBATIM)

set(LAYERNORM_GLSL ${SHADERS_DIR}/layernorm.comp.glsl)
set(LAYERNORM_SPV  ${CMAKE_CURRENT_BINARY_DIR}/layernorm.comp.spv)
add_custom_command(OUTPUT ${LAYERNORM_SPV}
  COMMAND ${GLSLANG_VALIDATOR} -V ${LAYERNORM_GLSL} -o ${LAYERNORM_SPV}
  DEPENDS ${LAYERNORM_GLSL}
  VERBATIM)

set(LAYERNORM_RMSNORM_FUSED_GLSL ${SHADERS_DIR}/layernorm_rmsnorm_fused.comp.glsl)
set(LAYERNORM_RMSNORM_FUSED_SPV  ${CMAKE_CURRENT_BINARY_DIR}/layernorm_rmsnorm_fused.comp.spv)
add_custom_command(OUTPUT ${LAYERNORM_RMSNORM_FUSED_SPV}
  COMMAND ${GLSLANG_VALIDATOR} -V ${LAYERNORM_RMSNORM_FUSED_GLSL} -o ${LAYERNORM_RMSNORM_FUSED_SPV}
  DEPENDS ${LAYERNORM_RMSNORM_FUSED_GLSL}
  VERBATIM)

set(LAYERNORM_RMSNORM_FUSED_TILED_GLSL ${SHADERS_DIR}/layernorm_rmsnorm_fused_tiled.comp.glsl)
set(LAYERNORM_RMSNORM_FUSED_TILED_SPV  ${CMAKE_CURRENT_BINARY_DIR}/layernorm_rmsnorm_fused_tiled.comp.spv)
add_custom_command(OUTPUT ${LAYERNORM_RMSNORM_FUSED_TILED_SPV}
  COMMAND ${GLSLANG_VALIDATOR} -V ${LAYERNORM_RMSNORM_FUSED_TILED_GLSL} -o ${LAYERNORM_RMSNORM_FUSED_TILED_SPV}
  DEPENDS ${LAYERNORM_RMSNORM_FUSED_TILED_GLSL}
  VERBATIM)

set(LAYERNORM_TILED_GLSL ${SHADERS_DIR}/layernorm_tiled.comp.glsl)
set(LAYERNORM_TILED_SPV  ${CMAKE_CURRENT_BINARY_DIR}/layernorm_tiled.comp.spv)
add_custom_command(OUTPUT ${LAYERNORM_TILED_SPV}
  COMMAND ${GLSLANG_VALIDATOR} -V ${LAYERNORM_TILED_GLSL} -o ${LAYERNORM_TILED_SPV}
  DEPENDS ${LAYERNORM_TILED_GLSL}
  VERBATIM)

set(RMSNORM_GLSL ${SHADERS_DIR}/rmsnorm.comp.glsl)
set(RMSNORM_SPV  ${CMAKE_CURRENT_BINARY_DIR}/rmsnorm.comp.spv)
add_custom_command(OUTPUT ${RMSNORM_SPV}
  COMMAND ${GLSLANG_VALIDATOR} -V ${RMSNORM_GLSL} -o ${RMSNORM_SPV}
  DEPENDS ${RMSNORM_GLSL}
  VERBATIM)

set(SOFTMAX_GLSL ${SHADERS_DIR}/softmax.comp.glsl)
set(SOFTMAX_SPV  ${CMAKE_CURRENT_BINARY_DIR}/softmax.comp.spv)
add_custom_command(OUTPUT ${SOFTMAX_SPV}
  COMMAND ${GLSLANG_VALIDATOR} -V ${SOFTMAX_GLSL} -o ${SOFTMAX_SPV}
  DEPENDS ${SOFTMAX_GLSL}
  VERBATIM)

set(RMSNORM_TILED_GLSL ${SHADERS_DIR}/rmsnorm_tiled.comp.glsl)
set(RMSNORM_TILED_SPV  ${CMAKE_CURRENT_BINARY_DIR}/rmsnorm_tiled.comp.spv)
add_custom_command(OUTPUT ${RMSNORM_TILED_SPV}
  COMMAND ${GLSLANG_VALIDATOR} -V ${RMSNORM_TILED_GLSL} -o ${RMSNORM_TILED_SPV}
  DEPENDS ${RMSNORM_TILED_GLSL}
  VERBATIM)

set(SOFTMAX_TILED_GLSL ${SHADERS_DIR}/softmax_tiled.comp.glsl)
set(SOFTMAX_TILED_SPV  ${CMAKE_CURRENT_BINARY_DIR}/softmax_tiled.comp.spv)
add_custom_command(OUTPUT ${SOFTMAX_TILED_SPV}
  COMMAND ${GLSLANG_VALIDATOR} -V ${SOFTMAX_TILED_GLSL} -o ${SOFTMAX_TILED_SPV}
  DEPENDS ${SOFTMAX_TILED_GLSL}
  VERBATIM)

add_custom_target(vk_shaders ALL
  DEPENDS
    ${FILL_SPV}
    ${GEMM_SPV}
    ${GEMM_TILED_SPV}
    ${GEMM_F16ACC32_SPV}
    ${GEMM_F16ACC32_VEC2_SPV}
    ${GEMM_F16ACC32_VEC2_32X8_SPV}
    ${GEMM_F16ACC32_VEC2_DB_SPV}
    ${GEMM_F16ACC32_SUBGROUP_SPV}
    ${LAYERNORM_SPV}
    ${LAYERNORM_RMSNORM_FUSED_SPV}
    ${LAYERNORM_RMSNORM_FUSED_TILED_SPV}
    ${LAYERNORM_TILED_SPV}
    ${RMSNORM_SPV}
    ${SOFTMAX_SPV}
    ${RMSNORM_TILED_SPV}
    ${SOFTMAX_TILED_SPV}
)

# -------------------------------------------------------------------
# Vulkan benches that use backend.cpp/buffer.cpp
add_executable(vk_fill_bench
  src/vk_fill_bench.cpp
  ../../../src/rt/backend/vulkan/src/backend.cpp
  ../../../src/rt/backend/vulkan/src/buffer.cpp
)
add_dependencies(vk_fill_bench vk_shaders)
target_compile_options(vk_fill_bench PRIVATE -O3 -march=native -pthread)
target_link_libraries(vk_fill_bench PRIVATE Vulkan::Vulkan)

add_executable(vk_gemm_bench
  src/vk_gemm_bench.cpp
  ../../../src/rt/backend/vulkan/src/backend.cpp
  ../../../src/rt/backend/vulkan/src/buffer.cpp
)
add_dependencies(vk_gemm_bench vk_shaders)
target_compile_options(vk_gemm_bench PRIVATE -O3 -march=native -pthread)
target_link_libraries(vk_gemm_bench PRIVATE Vulkan::Vulkan)

add_executable(vk_gemm_tiled_bench
  src/vk_gemm_tiled_bench.cpp
  ../../../src/rt/backend/vulkan/src/backend.cpp
  ../../../src/rt/backend/vulkan/src/buffer.cpp
)
add_dependencies(vk_gemm_tiled_bench vk_shaders)
target_compile_options(vk_gemm_tiled_bench PRIVATE -O3 -march=native -pthread)
target_link_libraries(vk_gemm_tiled_bench PRIVATE Vulkan::Vulkan)

add_executable(vk_gemm_tiled_ts_bench
  src/vk_gemm_tiled_ts_bench.cpp
  ../../../src/rt/backend/vulkan/src/backend.cpp
  ../../../src/rt/backend/vulkan/src/buffer.cpp
)
add_dependencies(vk_gemm_tiled_ts_bench vk_shaders)
target_compile_options(vk_gemm_tiled_ts_bench PRIVATE -O3 -march=native -pthread)
target_link_libraries(vk_gemm_tiled_ts_bench PRIVATE Vulkan::Vulkan)

add_executable(vk_gemm_f16acc32_tiled_ts_bench
  src/vk_gemm_f16acc32_tiled_ts_bench.cpp
  ../../../src/rt/backend/vulkan/src/backend.cpp
  ../../../src/rt/backend/vulkan/src/buffer.cpp
)
add_dependencies(vk_gemm_f16acc32_tiled_ts_bench vk_shaders)
target_compile_options(vk_gemm_f16acc32_tiled_ts_bench PRIVATE -O3 -march=native -pthread)
target_link_libraries(vk_gemm_f16acc32_tiled_ts_bench PRIVATE Vulkan::Vulkan)

add_executable(vk_gemm_f16acc32_tiled_vec2_ts_bench
  src/vk_gemm_f16acc32_tiled_vec2_ts_bench.cpp
  ../../../src/rt/backend/vulkan/src/backend.cpp
  ../../../src/rt/backend/vulkan/src/buffer.cpp
)
add_dependencies(vk_gemm_f16acc32_tiled_vec2_ts_bench vk_shaders)
target_compile_options(vk_gemm_f16acc32_tiled_vec2_ts_bench PRIVATE -O3 -march=native -pthread)
target_link_libraries(vk_gemm_f16acc32_tiled_vec2_ts_bench PRIVATE Vulkan::Vulkan)

add_executable(vk_gemm_f16acc32_tiled_vec2_32x8_ts_bench
  src/vk_gemm_f16acc32_tiled_vec2_32x8_ts_bench.cpp
  ../../../src/rt/backend/vulkan/src/backend.cpp
  ../../../src/rt/backend/vulkan/src/buffer.cpp
)
add_dependencies(vk_gemm_f16acc32_tiled_vec2_32x8_ts_bench vk_shaders)
target_compile_options(vk_gemm_f16acc32_tiled_vec2_32x8_ts_bench PRIVATE -O3 -march=native -pthread)
target_link_libraries(vk_gemm_f16acc32_tiled_vec2_32x8_ts_bench PRIVATE Vulkan::Vulkan)

add_executable(vk_gemm_f16acc32_tiled_vec2_db_ts_bench
  src/vk_gemm_f16acc32_tiled_vec2_db_ts_bench.cpp
  ../../../src/rt/backend/vulkan/src/backend.cpp
  ../../../src/rt/backend/vulkan/src/buffer.cpp
)
add_dependencies(vk_gemm_f16acc32_tiled_vec2_db_ts_bench vk_shaders)
target_compile_options(vk_gemm_f16acc32_tiled_vec2_db_ts_bench PRIVATE -O3 -march=native -pthread)
target_link_libraries(vk_gemm_f16acc32_tiled_vec2_db_ts_bench PRIVATE Vulkan::Vulkan)

add_executable(vk_layernorm_bench
  src/vk_layernorm_bench.cpp
  ../../../src/rt/backend/vulkan/src/backend.cpp
  ../../../src/rt/backend/vulkan/src/buffer.cpp
)
add_dependencies(vk_layernorm_bench vk_shaders)
target_compile_options(vk_layernorm_bench PRIVATE -O3 -march=native -pthread)
target_link_libraries(vk_layernorm_bench PRIVATE Vulkan::Vulkan)

add_executable(vk_layernorm_rmsnorm_fused_bench
  src/vk_layernorm_rmsnorm_fused_bench.cpp
  ../../../src/rt/backend/vulkan/src/backend.cpp
  ../../../src/rt/backend/vulkan/src/buffer.cpp
)
add_dependencies(vk_layernorm_rmsnorm_fused_bench vk_shaders)
target_compile_options(vk_layernorm_rmsnorm_fused_bench PRIVATE -O3 -march=native -pthread)
target_link_libraries(vk_layernorm_rmsnorm_fused_bench PRIVATE Vulkan::Vulkan)

add_executable(vk_layernorm_rmsnorm_fused_tiled_bench
  src/vk_layernorm_rmsnorm_fused_tiled_bench.cpp
  ../../../src/rt/backend/vulkan/src/backend.cpp
  ../../../src/rt/backend/vulkan/src/buffer.cpp
)
add_dependencies(vk_layernorm_rmsnorm_fused_tiled_bench vk_shaders)
target_compile_options(vk_layernorm_rmsnorm_fused_tiled_bench PRIVATE -O3 -march=native -pthread)
target_link_libraries(vk_layernorm_rmsnorm_fused_tiled_bench PRIVATE Vulkan::Vulkan)

add_executable(vk_layernorm_tiled_bench
  src/vk_layernorm_tiled_bench.cpp
  ../../../src/rt/backend/vulkan/src/backend.cpp
  ../../../src/rt/backend/vulkan/src/buffer.cpp
)
add_dependencies(vk_layernorm_tiled_bench vk_shaders)
target_compile_options(vk_layernorm_tiled_bench PRIVATE -O3 -march=native -pthread)
target_link_libraries(vk_layernorm_tiled_bench PRIVATE Vulkan::Vulkan)

add_executable(vk_rmsnorm_bench
  src/vk_rmsnorm_bench.cpp
  ../../../src/rt/backend/vulkan/src/backend.cpp
  ../../../src/rt/backend/vulkan/src/buffer.cpp
)
add_dependencies(vk_rmsnorm_bench vk_shaders)
target_compile_options(vk_rmsnorm_bench PRIVATE -O3 -march=native -pthread)
target_link_libraries(vk_rmsnorm_bench PRIVATE Vulkan::Vulkan)

add_executable(vk_softmax_bench
  src/vk_softmax_bench.cpp
  ../../../src/rt/backend/vulkan/src/backend.cpp
  ../../../src/rt/backend/vulkan/src/buffer.cpp
)
add_dependencies(vk_softmax_bench vk_shaders)
target_compile_options(vk_softmax_bench PRIVATE -O3 -march=native -pthread)
target_link_libraries(vk_softmax_bench PRIVATE Vulkan::Vulkan)

add_executable(vk_rmsnorm_tiled_bench
  src/vk_rmsnorm_tiled_bench.cpp
  ../../../src/rt/backend/vulkan/src/backend.cpp
  ../../../src/rt/backend/vulkan/src/buffer.cpp
)
add_dependencies(vk_rmsnorm_tiled_bench vk_shaders)
target_compile_options(vk_rmsnorm_tiled_bench PRIVATE -O3 -march=native -pthread)
target_link_libraries(vk_rmsnorm_tiled_bench PRIVATE Vulkan::Vulkan)

add_executable(vk_softmax_tiled_bench
  src/vk_softmax_tiled_bench.cpp
  ../../../src/rt/backend/vulkan/src/backend.cpp
  ../../../src/rt/backend/vulkan/src/buffer.cpp
)
add_dependencies(vk_softmax_tiled_bench vk_shaders)
target_compile_options(vk_softmax_tiled_bench PRIVATE -O3 -march=native -pthread)
target_link_libraries(vk_softmax_tiled_bench PRIVATE Vulkan::Vulkan)

add_executable(vk_gemm_f16acc32_subgroup_ts_bench
  src/vk_gemm_f16acc32_subgroup_ts_bench.cpp
)
add_dependencies(vk_gemm_f16acc32_subgroup_ts_bench vk_shaders)
target_compile_options(vk_gemm_f16acc32_subgroup_ts_bench PRIVATE -O3 -march=native -pthread)
target_link_libraries(vk_gemm_f16acc32_subgroup_ts_bench PRIVATE Vulkan::Vulkan)

# -------------------------------------------------------------------
# Runtime smoke test (uses GEMM runtime)
add_executable(vk_gemm_runtime_smoke
  src/vk_gemm_runtime_smoke.cpp
  ../../../src/rt/backend/vulkan/src/backend.cpp
  ../../../src/rt/backend/vulkan/src/buffer.cpp
  ../../../src/rt/backend/vulkan/src/gemm.cpp
  ../../../src/rt/backend/vulkan/kernels/gemm_f16acc32_runtime.cpp
  ../../../src/rt/backend/vulkan/kernels/gemm_f32_runtime.cpp
  ../../../src/rt/backend/vulkan/autotune/vk_autotune.cpp
)
add_dependencies(vk_gemm_runtime_smoke vk_shaders)
target_compile_options(vk_gemm_runtime_smoke PRIVATE -O3 -march=native -pthread)
target_link_libraries(vk_gemm_runtime_smoke PRIVATE Vulkan::Vulkan)

# -------------------------------------------------------------------
# Autotune module (OBJECT library)
set(VK_AUTOTUNE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../src/rt/backend/vulkan/autotune)

add_library(vk_autotune_obj OBJECT
  ${VK_AUTOTUNE_DIR}/vk_autotune.cpp
)
target_compile_options(vk_autotune_obj PRIVATE -O3 -march=native -pthread)
target_link_libraries(vk_autotune_obj PRIVATE Vulkan::Vulkan)

# Export include dir so executables can do: #include "vk_autotune.hpp"
target_include_directories(vk_autotune_obj PUBLIC
  ${VK_AUTOTUNE_DIR}
)

# Auto GEMM bench (wrapper that executes candidates + caches winner)
add_executable(vk_gemm_auto_ts_bench
  src/vk_gemm_auto_ts_bench.cpp
  ../../../src/rt/backend/vulkan/src/backend.cpp
  $<TARGET_OBJECTS:vk_autotune_obj>
)
add_dependencies(vk_gemm_auto_ts_bench vk_shaders)
target_compile_options(vk_gemm_auto_ts_bench PRIVATE -O3 -march=native -pthread)
target_include_directories(vk_gemm_auto_ts_bench PRIVATE
  ${VK_AUTOTUNE_DIR}
)
target_link_libraries(vk_gemm_auto_ts_bench PRIVATE Vulkan::Vulkan)
