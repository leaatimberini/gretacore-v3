# Phase 3: LLM Inference Pipeline

## Objective
Execute end-to-end inference of an LLM model (Llama-2-7B) using the native HIP backend on MI300X, with tokenization and autoregressive text generation.

## Key Deliverables
1. **Block Scheduler:** Execute N transformer layers sequentially.
2. **Weight Loader:** Read tensors from SafeTensors/GGUF format.
3. **Tokenizer:** Integrate SentencePiece or BPE for text I/O.
4. **Autoregressive Generation:** Token-by-token prediction loop with sampling.
5. **E2E Benchmarks:** Tokens/s and latency metrics.

## Phase 2 Dependencies
This phase directly depends on components validated in Phase 2:
- **MFMA GEMM:** 13 TFLOPS validated.
- **RMSNorm, Softmax:** Error < 1e-6.
- **RoPE, Causal Mask:** Validated in graph.
- **KV-Cache:** O(1) updates with bit-perfect fidelity.
- **Graph Runner:** Deterministic execution of kernel chains.

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     greta_infer (CLI)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐      │
│  │ Tokenizer   │  │ Generator    │  │ BlockScheduler     │      │
│  │  (BPE/SP)   │→ │  (Autoregr.) │→ │  (N layers graph)  │      │
│  └─────────────┘  └──────────────┘  └────────────────────┘      │
│                                               │                 │
│  ┌─────────────────────────────────┐          │                 │
│  │ WeightLoader                    │←─────────┘                 │
│  │  (SafeTensors / GGUF)           │                            │
│  └─────────────────────────────────┘                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               HIP Backend (gcore::rt::hip)                      │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────────┐     │
│  │ GEMM   │ │RMSNorm │ │ RoPE   │ │ SiLU   │ │ KV-Cache   │     │
│  │ MFMA   │ │        │ │        │ │        │ │            │     │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    AMD Instinct MI300X
```

## Implementation Strategy
1. **Week 1:** Weight Loader + Model Config + base structures.
2. **Week 2:** Block Scheduler + buffer pool + multi-layer execution.
3. **Week 3:** Tokenizer + Generator + autoregressive loop.
4. **Week 4:** CLI + E2E test + initial optimization.

## Success Criteria
- Execute `greta_infer` with real Llama-2-7B weights.
- Generate coherent text (>50 tokens).
- Measure tokens/s >= 10 tok/s (conservative baseline).
- No crashes or memory leaks.

## Risks and Mitigations
| Risk | Mitigation |
| :--- | :--- |
| Complex weight formats | Start with GGUF (simpler) |
| Memory pressure on MI300X | Pre-allocate all buffers at startup |
| Low initial throughput | Optimize after validating correctness |

## Owner
**Phase 3 Owner:** Leandro Emanuel Timberini
