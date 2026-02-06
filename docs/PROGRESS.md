# GRETA Core Progress Index

## Sync Status (2026-02-06)
- **Local HEAD**: `8c73f96` ✅
- **GitHub HEAD**: `8c73f96` ✅
- **Remote MI300X**: `8c73f96` ✅
- **AMD Reports**: 40 documents in `docs/AMD/`
- **Artifacts Remote**: All rescued to `artifacts_remote/_rescued_from_remote/`

## Phase Index

| Phase | Date | HEAD Hash | Objective | Root Cause | Result | Artifacts | AMD Report |
|-------|------|-----------|-----------|------------|--------|-----------|------------|
| B3.59 | 2026-02-05 | `8c73f96` | Embedding/DebugInput audit | CLEAN | Confirmed OK | [B3.59](artifacts_remote/2026-02-05/b3_59/) | [AMD_B3_59](docs/AMD/2026_02_05_B3_59_embedding_debug_input_audit.md) |
| B3.58 | 2026-02-05 | `a04fbc7` | RMSNorm wiring audit | `UPSTREAM_X_MISMATCH` | X0/Ceros en decode0 | [B3.58](artifacts_remote/2026-02-04/b3_58/) | N/A |
| B3.57.1 | 2026-02-04 | `a04fbc7` | RMSNorm divergence | `NORMOUT_SELECTION` | Confirmed | N/A | N/A |

---

## Complete AMD Report Index (40 documents)

| # | Report | Date | Status |
|---|--------|------|--------|
| 1 | B3.5 - Layer Trace Root Cause | 2026-02-03 | ✅ |
| 2 | B3.6 - Decode Readout Landscape | 2026-02-03 | ✅ |
| 3 | B3.6_rerun - Decode Readout Landscape | 2026-02-03 | ✅ |
| 4 | B3.7 - Analysis Decode Landscape | 2026-02-03 | ✅ |
| 5 | B3.8 - Embedding Layout Verification | 2026-02-03 | ✅ |
| 6 | B3.9 - Embedding Row Major Fix | 2026-02-03 | ✅ |
| 7 | B3.10 - Attractor Validation | 2026-02-03 | ✅ |
| 8 | B3.11 - Readout Consistency Fix | 2026-02-03 | ✅ |
| 9 | B3.12 - Decode Readout Semantics | 2026-02-03 | ✅ |
| 10 | B3.13 - Prefill/Decode Delta LMHead RMS | 2026-02-03 | ✅ |
| 11 | B3.14 - LMHead Force Route Isolation | 2026-02-03 | ✅ |
| 12 | B3.15 - LMHead Weight Layout Verify | 2026-02-03 | ✅ |
| 13 | B3.16 - LMHead MFMA Fix Acceptance | 2026-02-03 | ✅ |
| 14 | B3.17 - Decode LMHead Isolation | 2026-02-03 | ✅ |
| 15 | B3.18 - Prefill/Decode Hidden Equivalence | 2026-02-03 | ✅ |
| 16 | B3.19 - Decode Collapse Fix Acceptance | 2026-02-03 | ✅ |
| 17 | B3.20 - Attention Decode Isolation | 2026-02-03 | ✅ |
| 18 | B3.21 - Attention Decode MFMA Fix | 2026-02-03 | ✅ |
| 19 | B3.22 - Attention Precision Root Cause | 2026-02-03 | ✅ |
| 20 | B3.23 - Softmax Isolation Decode0 | 2026-02-03 | ✅ |
| 21 | B3.24 - V Accumulation Isolation | 2026-02-03 | ✅ |
| 22 | B3.25 - V Layout Fix Acceptance | 2026-02-03 | ✅ |
| 23 | B3.26 - V Addressing Long Context Fix | 2026-02-03 | ✅ |
| 24 | B3.27 - Decode0 Post-Attention Collapse | 2026-02-03 | ✅ |
| 25 | B3.28 - Decode0 Input Semantics Fix | 2026-02-03 | ✅ |
| 26 | B3.29 - Stage Trace Post-XIN Isolation | 2026-02-03 | ✅ |
| 27 | B3.30 - Layer0 Attention Pipeline Root Cause | 2026-02-03 | ✅ |
| 28 | B3.31 - Decode QKV Projection Route | 2026-02-03 | ✅ |
| 29 | B3.32 - Normout QKV Weight Probe | 2026-02-03 | ✅ |
| 30 | B3.33 - QKV Weight Layout Verification | 2026-02-03 | ✅ |
| 31 | B3.34 - Prefill WQ Layout Fix | 2026-02-03 | ✅ |
| 32 | B3.35 - Prefill WK Layout Fix | 2026-02-03 | ✅ |
| 33 | B3.36 - Prefill WV Layout Fix | 2026-02-03 | ✅ |
| 34 | B3.37 - Full Pipeline Acceptance | 2026-02-03 | ✅ |
| 35 | B3.38 - Post-QKV Root Cause Isolation | 2026-02-03 | ✅ |
| 36 | B3.39 - WO vs Residual Add Isolation | 2026-02-03 | ✅ |
| 37 | B3.40 - WO Projection Layout Fix | 2026-02-03 | ✅ |
| 38 | B3.41 - Post-WO Collapse Isolation | 2026-02-03 | ✅ |
| 39 | B3.42 - FFN RMSNorm Root Cause | 2026-02-03 | ✅ |
| 40 | B3.59 - Embedding Debug Input Audit | 2026-02-05 | ✅ |

---

## Technical Details (B3.59)
- **Objective**: Identify why `embedding_out/x_in` was reported zeroed. Resolved ambiguity using standardized `StageTrace` metadata (`token_id`, `route`).
- **Flags used**: `GRETA_TRACE_STAGE=1`, `GRETA_TRACE_STAGE_DEBUG_INPUT=1`.
- **Result**: **NO ZEROING found**. Perfect hash match between prefill/decode.
