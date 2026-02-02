CXX = g++
HIPCC = /opt/rocm/bin/hipcc
ROCM_PATH ?= /opt/rocm

# Attempt to find ROCM_PATH if it's a symbolic link or if it's in a versioned directory
ifeq ($(wildcard $(ROCM_PATH)),)
  ROCM_PATH = $(shell dirname $(shell which hipcc))/..
endif

INCLUDES = -I../../src/inference/include -I../../src/rt/include -I../../src/rt/backend/hip/include -I../../src/compute/include -I$(ROCM_PATH)/include

CXXFLAGS = -O3 -std=c++17 -fopenmp $(INCLUDES) -DGCORE_USE_HIP=1 -D__HIP_PLATFORM_AMD__=1
HIPFLAGS = -O3 --offload-arch=gfx942 $(INCLUDES)

LDFLAGS = -L$(ROCM_PATH)/lib -lamdhip64 -lhsa-runtime64 -fopenmp

# Objects needed from the core
OBJS = ../../src/inference/src/weight_loader.cpp \
       ../../src/rt/backend/hip/src/buffer.cpp \
       ../../src/rt/backend/hip/src/greta_runtime_hip.cpp

TARGET = greta_quantize_gguf

all: $(TARGET)

$(TARGET): greta_quantize_gguf.cpp $(OBJS)
	$(HIPCC) $(CXXFLAGS) greta_quantize_gguf.cpp $(OBJS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)
