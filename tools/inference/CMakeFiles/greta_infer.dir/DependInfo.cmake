
# Consider dependencies only in project.
set(CMAKE_DEPENDS_IN_PROJECT_ONLY OFF)

# The set of languages for which implicit dependencies are needed:
set(CMAKE_DEPENDS_LANGUAGES
  "HIP"
  )
# The set of files for implicit dependencies of each language:
set(CMAKE_DEPENDS_CHECK_HIP
  "/media/leandro/D08A27808A2762683/gretacore/gretacore/src/rt/backend/hip/kernels/attention_kernels.hip" "/media/leandro/D08A27808A2762683/gretacore/gretacore/tools/inference/CMakeFiles/greta_infer.dir/media/leandro/D08A27808A2762683/gretacore/gretacore/src/rt/backend/hip/kernels/attention_kernels.hip.o"
  "/media/leandro/D08A27808A2762683/gretacore/gretacore/src/rt/backend/hip/kernels/basic_kernels.hip" "/media/leandro/D08A27808A2762683/gretacore/gretacore/tools/inference/CMakeFiles/greta_infer.dir/media/leandro/D08A27808A2762683/gretacore/gretacore/src/rt/backend/hip/kernels/basic_kernels.hip.o"
  "/media/leandro/D08A27808A2762683/gretacore/gretacore/src/rt/backend/hip/kernels/fused_attention_kernels.hip" "/media/leandro/D08A27808A2762683/gretacore/gretacore/tools/inference/CMakeFiles/greta_infer.dir/media/leandro/D08A27808A2762683/gretacore/gretacore/src/rt/backend/hip/kernels/fused_attention_kernels.hip.o"
  "/media/leandro/D08A27808A2762683/gretacore/gretacore/src/rt/backend/hip/kernels/fused_compute_kernels.hip" "/media/leandro/D08A27808A2762683/gretacore/gretacore/tools/inference/CMakeFiles/greta_infer.dir/media/leandro/D08A27808A2762683/gretacore/gretacore/src/rt/backend/hip/kernels/fused_compute_kernels.hip.o"
  "/media/leandro/D08A27808A2762683/gretacore/gretacore/src/rt/backend/hip/kernels/gemm_kernels.hip" "/media/leandro/D08A27808A2762683/gretacore/gretacore/tools/inference/CMakeFiles/greta_infer.dir/media/leandro/D08A27808A2762683/gretacore/gretacore/src/rt/backend/hip/kernels/gemm_kernels.hip.o"
  )
set(CMAKE_HIP_COMPILER_ID "Clang")

# Preprocessor definitions for this target.
set(CMAKE_TARGET_DEFINITIONS_HIP
  "GCORE_USE_HIP=1"
  "__HIP_PLATFORM_AMD__=1"
  "__HIP_ROCclr__=1"
  )

# The include file search paths:
set(CMAKE_HIP_TARGET_INCLUDE_PATH
  "../../src/inference/include"
  "../../src/rt/backend/hip/include"
  "../../src/rt/include"
  "../../src/compute/include"
  )

# The set of dependency files which are needed:
set(CMAKE_DEPENDS_DEPENDENCY_FILES
  "/media/leandro/D08A27808A2762683/gretacore/gretacore/src/compute/src/greta_compute_hip.cpp" "CMakeFiles/greta_infer.dir/media/leandro/D08A27808A2762683/gretacore/gretacore/src/compute/src/greta_compute_hip.cpp.o" "gcc" "CMakeFiles/greta_infer.dir/media/leandro/D08A27808A2762683/gretacore/gretacore/src/compute/src/greta_compute_hip.cpp.o.d"
  "/media/leandro/D08A27808A2762683/gretacore/gretacore/src/inference/src/block_scheduler.cpp" "CMakeFiles/greta_infer.dir/media/leandro/D08A27808A2762683/gretacore/gretacore/src/inference/src/block_scheduler.cpp.o" "gcc" "CMakeFiles/greta_infer.dir/media/leandro/D08A27808A2762683/gretacore/gretacore/src/inference/src/block_scheduler.cpp.o.d"
  "/media/leandro/D08A27808A2762683/gretacore/gretacore/src/inference/src/generator.cpp" "CMakeFiles/greta_infer.dir/media/leandro/D08A27808A2762683/gretacore/gretacore/src/inference/src/generator.cpp.o" "gcc" "CMakeFiles/greta_infer.dir/media/leandro/D08A27808A2762683/gretacore/gretacore/src/inference/src/generator.cpp.o.d"
  "/media/leandro/D08A27808A2762683/gretacore/gretacore/src/inference/src/tokenizer.cpp" "CMakeFiles/greta_infer.dir/media/leandro/D08A27808A2762683/gretacore/gretacore/src/inference/src/tokenizer.cpp.o" "gcc" "CMakeFiles/greta_infer.dir/media/leandro/D08A27808A2762683/gretacore/gretacore/src/inference/src/tokenizer.cpp.o.d"
  "/media/leandro/D08A27808A2762683/gretacore/gretacore/src/inference/src/weight_loader.cpp" "CMakeFiles/greta_infer.dir/media/leandro/D08A27808A2762683/gretacore/gretacore/src/inference/src/weight_loader.cpp.o" "gcc" "CMakeFiles/greta_infer.dir/media/leandro/D08A27808A2762683/gretacore/gretacore/src/inference/src/weight_loader.cpp.o.d"
  "/media/leandro/D08A27808A2762683/gretacore/gretacore/src/rt/backend/hip/src/buffer.cpp" "CMakeFiles/greta_infer.dir/media/leandro/D08A27808A2762683/gretacore/gretacore/src/rt/backend/hip/src/buffer.cpp.o" "gcc" "CMakeFiles/greta_infer.dir/media/leandro/D08A27808A2762683/gretacore/gretacore/src/rt/backend/hip/src/buffer.cpp.o.d"
  "/media/leandro/D08A27808A2762683/gretacore/gretacore/src/rt/backend/hip/src/greta_runtime_hip.cpp" "CMakeFiles/greta_infer.dir/media/leandro/D08A27808A2762683/gretacore/gretacore/src/rt/backend/hip/src/greta_runtime_hip.cpp.o" "gcc" "CMakeFiles/greta_infer.dir/media/leandro/D08A27808A2762683/gretacore/gretacore/src/rt/backend/hip/src/greta_runtime_hip.cpp.o.d"
  "/media/leandro/D08A27808A2762683/gretacore/gretacore/tools/inference/src/greta_infer.cpp" "CMakeFiles/greta_infer.dir/src/greta_infer.cpp.o" "gcc" "CMakeFiles/greta_infer.dir/src/greta_infer.cpp.o.d"
  )

# Targets to which this target links which contain Fortran sources.
set(CMAKE_Fortran_TARGET_LINKED_INFO_FILES
  )

# Targets to which this target links which contain Fortran sources.
set(CMAKE_Fortran_TARGET_FORWARD_LINKED_INFO_FILES
  )

# Fortran module output directory.
set(CMAKE_Fortran_TARGET_MODULE_DIR "")
