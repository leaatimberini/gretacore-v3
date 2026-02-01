#!/bin/bash
# GRETA CORE - MI300X Vulkan Diagnostic Script

LOG_FILE="tools/bench/runtime/results/$(date +%Y-%m-%d)_mi300x_diag.log"
echo "--- MI300X VULKAN DIAGNOSTIC ---" | tee -a $LOG_FILE
date | tee -a $LOG_FILE

echo -e "\n1. System Info" | tee -a $LOG_FILE
uname -a | tee -a $LOG_FILE
lsb_release -a | tee -a $LOG_FILE

echo -e "\n2. Vulkan Drivers Found" | tee -a $LOG_FILE
ls /usr/share/vulkan/icd.d/ | tee -a $LOG_FILE

echo -e "\n3. Vulkan Device List" | tee -a $LOG_FILE
vulkaninfo --summary | tee -a $LOG_FILE

echo -e "\n4. ROCm Version" | tee -a $LOG_FILE
if command -v rocm-smi &> /dev/null; then
    rocm-smi --showproductname --showversion | tee -a $LOG_FILE
else
    echo "rocm-smi not found" | tee -a $LOG_FILE
fi

echo -e "\n5. Checking for Known MI300X / RADV Issues" | tee -a $LOG_FILE
dmesg | grep -i "amdgpu" | tail -n 20 | tee -a $LOG_FILE

echo -e "\n6. Kernel Dispatch Test (Minimal)" | tee -a $LOG_FILE
# Intentar correr el bench más básico que haya fallado
if [ -f "tools/bench/runtime/build/vk_smoke_bench" ]; then
    ./tools/bench/runtime/build/vk_smoke_bench --iters 1 2>&1 | tee -a $LOG_FILE
else
    echo "vk_smoke_bench not found. Run build first." | tee -a $LOG_FILE
fi

echo -e "\n--- End of Diagnostic ---" | tee -a $LOG_FILE
echo "Log saved to $LOG_FILE"
