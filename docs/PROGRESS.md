# GRETA Core Progress Index

## Sync Status (2026-02-06)
- **Local HEAD**: `ead8681` ✅
- **GitHub HEAD**: `ead8681` ✅
- **Remote MI300X**: `ead8681` ✅
- **AMD Reports**: 41 documents in `docs/AMD/`
- **Artifacts Remote**: All rescued to `artifacts_remote/_rescued_from_remote/`

## Phase Index

| Phase | Date | HEAD Hash | Objective | Root Cause | Result | Artifacts | AMD Report |
|-------|------|-----------|-----------|------------|--------|-----------|------------|
| B3.60 | 2026-02-06 | `ead8681` | Attention Block Bisect | PASS | 3/3 tokens OK | [B3.60](artifacts_remote/_rescued_from_remote/artifacts_remote/2026-02-06/b3_60/) | [AMD_B3_60](docs/AMD/2026_02_06_B3_60_attention_block_bisect_audit.md) |
| B3.59 | 2026-02-05 | `d558073` | Embedding/DebugInput audit | CLEAN | Confirmed OK | [B3.59](artifacts_remote/2026-02-05/b3_59/) | [AMD_B3_59](docs/AMD/2026_02_05_B3_59_embedding_debug_input_audit.md) |
| B3.58 | 2026-02-05 | `d558073` | RMSNorm wiring audit | `UPSTREAM_X_MISMATCH` | X0/Ceros en decode0 | [B3.58](artifacts_remote/2026-02-04/b3_58/) | N/A |
| B3.57.1 | 2026-02-04 | `d558073` | RMSNorm divergence | `NORMOUT_SELECTION` | Confirmed | N/A | N/A |

---

## Complete AMD Report Index (40 documents)

See [docs/AMD/INDEX.md](docs/AMD/INDEX.md) for full index with categories and links.

---

## Technical Details (B3.59)
- **Objective**: Identify why `embedding_out/x_in` was reported zeroed. Resolved ambiguity using standardized `StageTrace` metadata (`token_id`, `route`).
- **Flags used**: `GRETA_TRACE_STAGE=1`, `GRETA_TRACE_STAGE_DEBUG_INPUT=1`.
- **Result**: **NO ZEROING found**. Perfect hash match between prefill/decode.
