# GRETA CORE — Runtime Benchmarks

Path: tools/bench/runtime/README.md  
Version: 1.0  
Language: EN/ES (bilingual)

## EN
Benchmarks for GRETA CORE runtime components and LLM primitives.
- `llm_primitives_bench` (LayerNorm, RMSNorm, Softmax; CPU reference + checks)
- `vk_layernorm_bench` (Vulkan LayerNorm baseline + validation)
- `vk_layernorm_rmsnorm_fused_bench` (Vulkan LayerNorm+RMSNorm fused + validation)
- `vk_layernorm_rmsnorm_fused_tiled_bench` (Vulkan LayerNorm+RMSNorm fused tiled + validation)
- `vk_layernorm_tiled_bench` (Vulkan LayerNorm tiled + validation)
- `vk_rmsnorm_bench` (Vulkan RMSNorm baseline + validation)
- `vk_softmax_bench` (Vulkan Softmax baseline + validation)
- `vk_rmsnorm_tiled_bench` (Vulkan RMSNorm tiled + validation)
- `vk_softmax_tiled_bench` (Vulkan Softmax tiled + validation)
Presets (local/remote):
- `tools/bench/runtime/scripts/run_presets_local.sh smoke|standard|perf|verify` (includes LLM runs)
- `tools/bench/runtime/scripts/run_presets_remote.sh user@host /tmp/greta smoke|standard|perf|verify`

## ES
Benchmarks para componentes del runtime de GRETA CORE y primitivas LLM.
- `llm_primitives_bench` (LayerNorm, RMSNorm, Softmax; referencia CPU + checks)
- `vk_layernorm_bench` (baseline Vulkan de LayerNorm + validación)
- `vk_layernorm_rmsnorm_fused_bench` (Vulkan LayerNorm+RMSNorm fused + validación)
- `vk_layernorm_rmsnorm_fused_tiled_bench` (Vulkan LayerNorm+RMSNorm fused tiled + validación)
- `vk_layernorm_tiled_bench` (Vulkan LayerNorm tiled + validación)
- `vk_rmsnorm_bench` (baseline Vulkan de RMSNorm + validación)
- `vk_softmax_bench` (baseline Vulkan de Softmax + validación)
- `vk_rmsnorm_tiled_bench` (Vulkan RMSNorm tiled + validación)
- `vk_softmax_tiled_bench` (Vulkan Softmax tiled + validación)
Presets (local/remoto):
- `tools/bench/runtime/scripts/run_presets_local.sh smoke|standard|perf|verify` (incluye LLM)
- `tools/bench/runtime/scripts/run_presets_remote.sh user@host /tmp/greta smoke|standard|perf|verify`
