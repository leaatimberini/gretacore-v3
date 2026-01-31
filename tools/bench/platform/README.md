# GRETA CORE — Platform Benchmarks

Path: tools/bench/platform/README.md  
Version: 1.0  
Language: EN/ES (single file; bilingual sections)

## Purpose
These benchmarks measure platform/hardware limits independently of GRETA CORE runtime:
- CPU memory bandwidth (DDR5 throughput)
- GPU kernel launch overhead (HIP no-op) when available
- GPU vector add bandwidth (HIP) when available

## Build (Ubuntu 22.04)
From repo root:

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build linux-tools-common linux-tools-generic
