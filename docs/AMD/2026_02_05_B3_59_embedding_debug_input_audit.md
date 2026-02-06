# AMD Audit Report: B3.59 Embedding + StageDebugInput

**Date**: 2026-02-05
**Auditor**: Leandro Emanuel Timberini (L.E.T)
**Platform**: AMD Instinct MI300X

## 1. Objective
Audit the `embedding_out/x_in` point in the `decode0` phase when `GRETA_TRACE_STAGE_DEBUG_INPUT` is enabled. Resolve the ambiguity of `UPSTREAM_X_MISMATCH` by verifying if the embedding lookup itself is zeroed or divergent on the MI300X.

## 2. Methodology
- **Instrumentation**: Added `token_id` and `route` metadata to `StageTrace` JSON output. Standardized metadata capture for all trace points.
- **Debug Injection**: Used `GRETA_TRACE_STAGE_DEBUG_INPUT=1` to inject a target token into the hidden buffer override.
- **Consistency Check**: Compared hash and non-zero counts (`nz_count`) between the `prefill_last` lookup and the `decode0` injected input.
- **Weight Verification**: Captured a 1KB fingerprint of embedding weights (`embd_w_hash`) to ensure weight stability between phases.

## 3. Results (Summary)

| Token ID | Phase Pairing | Status | NZ Count (Sample 256) | Hash Match |
| :--- | :--- | :--- | :---: | :---: |
| 10 | Prefill vs Decode | **OK** | 256 / 256 | Yes |
| 108 | Prefill vs Decode | **OK** | 256 / 256 | Yes |

### Final Audit Table (Verifiable Artifacts)

|   token_id |           p_emb_hash |           d_emb_hash |           d_xin_hash |   d_xin_nz | route               | status   |
|-----------:|---------------------:|---------------------:|---------------------:|-----------:|:--------------------|:---------|
|         10 |  7648363339390251139 |  7648363339390251139 |  7648363339390251139 |        256 | EMBED_LOOKUP_DECODE | OK       |
|        108 | 12821427986147713091 | 12821427986147713091 | 12821427986147713091 |        256 | EMBED_LOOKUP_DECODE | OK       |

## 4. Conclusion
The B3.59 audit confirms that **no zeroing occurs at the embedding output/x_in point** on the MI300X when using the debug input mechanism. The perfect hash match between prefill and decode phases indicates that:
1. The lookup logic is consistent.
2. The weights are valid.
3. The injection into the hidden buffer (`x_in`) preserves data integrity.

Previous reports of zeroing in B3.58 were either pairing artifacts in the analyzer or originate further downstream in the pipeline (e.g., Attention or MLP blocks).

## 5. Artifact Index
- **TGZ**: `artifacts_remote/2026-02-05/b3_59/gretacore_b3_59_artifacts.tgz`
- **Analysis**: `artifacts_remote/2026-02-05/b3_59/b3_59_analysis.txt`
- **Hash (HEAD)**: `f681dc9938fcf0d230e52be2138c79010583fb67`

---
*Signed,*
**Leandro Emanuel Timberini**
L.E.T / Release Manager & Tech Lead
