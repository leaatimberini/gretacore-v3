# GRETA CORE AMD Reports Index

**Repository:** gretacore  
**Branch:** `b3_59_embedding_debug_input_audit`  
**Last Updated:** 2026-02-06  
**Total Reports:** 40

---

## Quick Navigation

| Category | Reports | Location |
|----------|---------|----------|
| All Reports | 40 | [reports/](./reports/) |
| Phase 3 | See below | [phases/PHASE3.md](./phases/PHASE3.md), [phases/PHASE3_ES.md](./phases/PHASE3_ES.md) |
| Progress | See below | [docs/PROGRESS.md](../PROGRESS.md) |

---

## Reports by Date

### 2026-02-05 (Latest)

| ID | Report | Status | Artifacts |
|----|--------|--------|-----------|
| B3.59 | Embedding/DebugInput Audit | ✅ PASS | [`2026-02-05/b3_59/`](../artifacts_remote/2026-02-05/b3_59/) |

### 2026-02-03 (Historical Analysis)

| ID | Report | Status | Focus Area |
|----|--------|--------|------------|
| B3.5 | Layer Trace Root Cause | ✅ PASS | Embedding/Layer Trace |
| B3.6 | Decode Readout Landscape | ✅ PASS | Decode Readout Analysis |
| B3.6_rerun | Decode Readout Landscape (Rerun) | ✅ PASS | Decode Readout Verification |
| B3.7 | Analysis Decode Landscape | ✅ PASS | Decode Pipeline Analysis |
| B3.8 | Embedding Layout Verification | ✅ PASS | Embedding Layout |
| B3.9 | Embedding Row Major Fix | ✅ PASS | Embedding Data Format |
| B3.10 | Attractor Validation | ✅ PASS | Attractor Behavior |
| B3.11 | Readout Consistency Fix | ✅ PASS | Readout Consistency |
| B3.12 | Decode Readout Semantics | ✅ PASS | Decode Readout Semantics |
| B3.13 | Prefill/Decode Delta LMHead RMS | ✅ PASS | LMHead RMS Analysis |
| B3.14 | LMHead Force Route Isolation | ✅ PASS | LMHead Route Isolation |
| B3.15 | LMHead Weight Layout Verify | ✅ PASS | LMHead Weight Layout |
| B3.16 | LMHead MFMA Fix Acceptance | ✅ PASS | LMHead MFMA Fix |
| B3.17 | Decode LMHead Isolation | ✅ PASS | Decode LMHead |
| B3.18 | Prefill/Decode Hidden Equivalence | ✅ PASS | Hidden State Equivalence |
| B3.19 | Decode Collapse Fix Acceptance | ✅ PASS | Decode Collapse Fix |
| B3.20 | Attention Decode Isolation | ✅ PASS | Attention Decode Isolation |
| B3.21 | Attention Decode MFMA Fix | ✅ PASS | Attention MFMA Fix |
| B3.22 | Attention Precision Root Cause | ✅ PASS | Attention Precision |
| B3.23 | Softmax Isolation Decode0 | ✅ PASS | Softmax Decode0 |
| B3.24 | V Accumulation Isolation | ✅ PASS | V Accumulation |
| B3.25 | V Layout Fix Acceptance | ✅ PASS | V Layout Fix |
| B3.26 | V Addressing Long Context Fix | ✅ PASS | V Addressing Long Context |
| B3.27 | Decode0 Post-Attention Collapse | ✅ PASS | Decode0 Post-Attention |
| B3.28 | Decode0 Input Semantics Fix | ✅ PASS | Decode0 Input Semantics |
| B3.29 | Stage Trace Post-XIN Isolation | ✅ PASS | Stage Trace Post-XIN |
| B3.30 | Layer0 Attention Pipeline Root Cause | ✅ PASS | Layer0 Attention Pipeline |
| B3.31 | Decode QKV Projection Route | ✅ PASS | QKV Projection Route |
| B3.32 | Normout QKV Weight Probe | ✅ PASS | Normout QKV Weight |
| B3.33 | QKV Weight Layout Verification | ✅ PASS | QKV Weight Layout |
| B3.34 | Prefill WQ Layout Fix | ✅ PASS | Prefill WQ Layout |
| B3.35 | Prefill WK Layout Fix | ✅ PASS | Prefill WK Layout |
| B3.36 | Prefill WV Layout Fix | ✅ PASS | Prefill WV Layout |
| B3.37 | Full Pipeline Acceptance | ✅ PASS | Full Pipeline |
| B3.38 | Post-QKV Root Cause Isolation | ✅ PASS | Post-QKV Root Cause |
| B3.39 | WO vs Residual Add Isolation | ✅ PASS | WO vs Residual Add |
| B3.40 | WO Projection Layout Fix | ✅ PASS | WO Projection Layout |
| B3.41 | Post-WO Collapse Isolation | ✅ PASS | Post-WO Collapse |
| B3.42 | FFN RMSNorm Root Cause | ✅ PASS | FFN RMSNorm |

---

## Reports by Category

### Core Documents
| Document | ES Version | Description |
|----------|------------|-------------|
| [phases/PHASE3.md](./phases/PHASE3.md) | [phases/PHASE3_ES.md](./phases/PHASE3_ES.md) | Executive summary and milestones |
| [phases/AMD_TECH_REPORT_EN.md](./phases/AMD_TECH_REPORT_EN.md) | [phases/AMD_TECH_REPORT_ES.md](./phases/AMD_TECH_REPORT_ES.md) | Full technical report |
| [phases/BENCH_PROTOCOL_EN.md](./phases/BENCH_PROTOCOL_EN.md) | [phases/BENCH_PROTOCOL_ES.md](./phases/BENCH_PROTOCOL_ES.md) | Benchmarking protocol |

### Reproduction Guides
| Document | ES Version | Description |
|----------|------------|-------------|
| [phases/PHASE3_REPRO_GUIDE_EN.md](./phases/PHASE3_REPRO_GUIDE_EN.md) | [phases/PHASE3_REPRO_GUIDE_ES.md](./phases/PHASE3_REPRO_GUIDE_ES.md) | Step-by-step reproduction |
| [phases/REPRO_CHECKLIST_EN.md](./phases/REPRO_CHECKLIST_EN.md) | [phases/REPRO_CHECKLIST_ES.md](./phases/REPRO_CHECKLIST_ES.md) | Verification checklist |

### Release Notes
| Document | ES Version | Description |
|----------|------------|-------------|
| [phases/RELEASE_NOTES_v0.1_EN.md](./phases/RELEASE_NOTES_v0.1_EN.md) | [phases/RELEASE_NOTES_v0.1_ES.md](./phases/RELEASE_NOTES_v0.1_ES.md) | v0.1 release notes |

### Schemas
| Document | Description |
|----------|-------------|
| [phases/bench_schema.csv](./phases/bench_schema.csv) | Benchmark schema definitions |

---

### Embedding
- B3.5, B3.8, B3.9, B3.59

### Attention
- B3.20, B3.21, B3.22, B3.23, B3.24, B3.25, B3.26, B3.30

### LMHead
- B3.13, B3.14, B3.15, B3.16, B3.17, B3.19

### QKV/Projection
- B3.31, B3.32, B3.33, B3.34, B3.35, B3.36, B3.38, B3.39, B3.40, B3.41

### FFN/Normalization
- B3.42

### Readout/Decode
- B3.6, B3.6_rerun, B3.7, B3.11, B3.12, B3.18, B3.27, B3.28, B3.29

### Full Pipeline
- B3.10, B3.37

---

## Artifacts Reference

| Artifact Location | Associated Reports | Status |
|-------------------|-------------------|--------|
| `artifacts_remote/2026-02-03/` | B3.5 - B3.42 | ✅ Available |
| `artifacts_remote/2026-02-04/` | B3.53 - B3.59 | ✅ Available |
| `artifacts_remote/2026-02-05/` | B3.59 | ✅ Available |

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [PROGRESS.md](../PROGRESS.md) | Complete progress index with technical details |
| [ROADMAP.md](../ROADMAP.md) | Project roadmap |
| [DEBUGGING.md](../DEBUGGING.md) | Debugging guidelines |
| [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) | How to reproduce AMD results |

---

*Maintained by: Leandro Emanuel Timberini*  
*Hardware: AMD MI300X*
