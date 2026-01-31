cmake_minimum_required(VERSION 3.22)

project(greta_platform_bench LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(membw_cpu src/membw_cpu.cpp)
target_compile_options(membw_cpu PRIVATE -O3 -march=native)
add_executable(memlat_cpu src/memlat_cpu.cpp)
target_compile_options(memlat_cpu PRIVATE -O3 -march=native)
# ---- HIP bench (opcional) ----
option(GRETA_ENABLE_HIP "Enable HIP benchmarks (requires ROCm/HIP)" ON)

if (GRETA_ENABLE_HIP)
  find_package(HIP QUIET)
  if (HIP_FOUND)
    add_executable(hip_noop_launch src/hip_noop_launch.cpp)
    target_compile_options(hip_noop_launch PRIVATE -O3)
    target_link_libraries(hip_noop_launch PRIVATE HIP::hip_runtime)
    target_compile_definitions(hip_noop_launch PRIVATE GRETA_HAS_HIP=1)
  else()
    message(STATUS "HIP no encontrado. hip_noop_launch no se va a compilar.")
  endif()
endif()
