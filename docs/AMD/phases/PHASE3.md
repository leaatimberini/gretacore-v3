# GRETA CORE Phase 3: AMD MI300X Optimization

**Start Date:** 2026-01-XX  
**Hardware:** AMD MI300X  
**Status:** IN PROGRESS  
**Current Branch:** `b3_59_embedding_debug_input_audit`

---

## Executive Summary

Phase 3 focuses on optimizing GRETA CORE for AMD MI300X hardware, addressing precision issues, and validating the full inference pipeline.

---

## Objectives

1. **Precision Validation**: Ensure float16 precision across all operations
2. **Full Pipeline Integration**: Validate end-to-end inference
3. **AMD Backend Optimization**: Leverage MI300X specific features
4. **Reproducibility**: Document all experiments and results

---

## Completed Milestones

| Milestone | B3 ID | Status | Date |
|-----------|-------|--------|------|
| Layer Trace Root Cause | B3.5 | ✅ DONE | 2026-02-03 |
| Decode Readout Analysis | B3.6 | ✅ DONE | 2026-02-03 |
| Embedding Layout Fix | B3.8, B3.9 | ✅ DONE | 2026-02-03 |
| LMHead Isolation | B3.13-B3.17 | ✅ DONE | 2026-02-03 |
| Attention Pipeline | B3.20-B3.30 | ✅ DONE | 2026-02-03 |
| QKV/Projection Fixes | B3.31-B3.36 | ✅ DONE | 2026-02-03 |
| Full Pipeline Acceptance | B3.37 | ✅ DONE | 2026-02-03 |
| FFN RMSNorm Root Cause | B3.42 | ✅ DONE | 2026-02-03 |
| Embedding Audit | B3.59 | ✅ DONE | 2026-02-05 |

---

## Current Focus (B3.59)

**Task:** Embedding/DebugInput Audit  
**Status:** ✅ COMPLETED  
**Result:** No zeroing found - perfect hash match between prefill/decode

### Technical Details
- **Flags Used:** `GRETA_TRACE_STAGE=1`, `GRETA_TRACE_STAGE_DEBUG_INPUT=1`
- **Issue:** Ambiguity in `embedding_out/x_in` reporting
- **Solution:** Standardized `StageTrace` metadata (`token_id`, `route`)

---

## Architecture Components Addressed

### 1. Embedding Layer ✅
- Layout verification (B3.8)
- Row major fix (B3.9)
- Debug input audit (B3.59)

### 2. Attention Mechanism ✅
- QKV projection isolation (B3.31)
- Attention decode isolation (B3.20)
- MFMA fix acceptance (B3.21)

### 3. LMHead ✅
- Route isolation (B3.14)
- Weight layout verification (B3.15)
- MFMA fix (B3.16)

### 4. Output Projections ✅
- WO projection layout (B3.40)
- Post-WO collapse isolation (B3.41)

### 5. FFN/Normalization ✅
- FFN RMSNorm root cause (B3.42)
- V addressing long context (B3.26)

---

## Next Steps

1. **B3.60**: TBD - Next optimization task
2. **Performance Benchmarking**: Full pipeline performance validation
3. **Documentation**: Complete Phase 3 technical report

---

## Key Artifacts

| Artifact | Location | Associated Reports |
|----------|----------|-------------------|
| B3.59 Analysis | `artifacts_remote/2026-02-05/b3_59/` | B3.59 |
| B3.42 Analysis | `artifacts_remote/2026-02-04/b3_58/` | B3.58 |
| B3.37 Pipeline | `artifacts_remote/2026-02-03/` | B3.37 |

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [AMD Reports Index](./INDEX.md) | All Phase 3 AMD reports |
| [PROGRESS.md](../PROGRESS.md) | Complete progress tracking |
| [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) | How to reproduce results |

---

## Hardware Specifications

| Component | Specification |
|-----------|---------------|
| GPU | AMD MI300X |
| Precision | float16 |
| Backend | HIP |

---

*Maintained by: Leandro Emanuel Timberini*  
*Last Updated: 2026-02-06*
