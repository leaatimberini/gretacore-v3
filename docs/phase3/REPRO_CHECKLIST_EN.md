# GRETA CORE - Reproducibility Checklist (Phase 3)

This checklist ensures that the technical evidence for Phase 3 is correctly gathered and verified.

## 1. Environment Verification
- [ ] ROCm version is 6.2 or 7.1.
- [ ] GPU is detected as `gfx942` (MI300X).
- [ ] `hipcc --version` returns expected compiler info.

## 2. Build Verification
- [ ] `cmake -DCMAKE_BUILD_TYPE=Release ..` completes without errors.
- [ ] `make -j` produces the `greta_infer` binary.
- [ ] System metadata appears when running with `GRETA_VERBOSE_INFO=1`.

## 3. Correctness Verification
- [ ] Prompt "Hi" generates 32 valid tokens.
- [ ] No NaNs/Infs are present in output.
- [ ] (Optional) `GRETA_TRACE_LEVEL=1` confirms stability across all layers.

## 4. Performance Verification
- [ ] Tokens/second matches or exceeds 12.5 T/s on MI300X.
- [ ] TTFT is under 250ms for short prompts.
- [ ] Running `./tools/phase3/run_bench.sh` produces a valid `bench_results.csv`.

## 5. Metadata & Registry
- [ ] Document hash recorded for legal/IP auditing.
- [ ] Reference GGUF model: `llama-2-7b-chat.Q4_K_M.gguf`.
