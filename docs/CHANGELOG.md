# GRETA Core Work Progress Changelog

## [2026-02-05] B3.59: Embedding + StageDebugInput Audit
- **Goal**: Audit `embedding_out` and `x_in` in `decode0` under `DEBUG_INPUT`.
- **Changes**:
  - Instrumented `StageTrace` with `token_id`, `nz_count`, and `route` labels.
  - Added `embed_out` probe in `BlockScheduler::forward`.
  - Added `GRETA_TRACE_DEBUG_INPUT_TOKEN_ID` environment bridge in `Generator::generate_tokens`.
  - Fixed remote build path issues by identifying correct `Make` targets in `tools/inference`.
- **Findings**: (TBD)

## [2026-02-05] B3.58: RMSNorm Wiring Audit
- **Goal**: Pinpoint divergence in RMSNorm.
- **Result**: Divergence confirmed to be caused by `UPSTREAM_X_MISMATCH`. Input `x` to RMSNorm is zero or invalid during `decode0` when `DEBUG_INPUT` is enabled.

## [2026-02-04] B3.57.1: RMSNorm Selection
- **Goal**: Fix logical token index mismatch.
- **Result**: `ROOT_CAUSE = NORMOUT_SELECTION`. Fixed.
