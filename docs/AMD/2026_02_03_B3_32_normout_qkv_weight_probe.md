# B3.32 — NormOut vs QKV Weight Probe (Layer0, Decode0)

## Resumen Ejecutivo (ES)
En B3.32 se instrumentó la traza de `attn_norm_in` y `attn_norm_out` en la proyección QKV de layer 0, comparando `prefill_last` vs `decode0` para cuatro prompts (short, p4_sys, p5_ba, long). Los resultados muestran que **`attn_norm_out` coincide (MAE=0.0) en todos los casos**, mientras que **`q` diverge fuertemente** en prompts con contexto (p4_sys/p5_ba/p6_long). Esto descarta a RMSNorm como causa primaria y apunta a un **problema de interpretación/packing de pesos QKV en el path de decode**, dependiente del contexto.

## Executive Summary (EN)
B3.32 instrumented `attn_norm_in`/`attn_norm_out` at layer-0 QKV projection and compared `prefill_last` vs `decode0` across four prompts (short, p4_sys, p5_ba, long). Results show **`attn_norm_out` matches (MAE=0.0) in all cases**, while **`q` diverges sharply** for prompts with context (p4_sys/p5_ba/p6_long). This rules out RMSNorm as the primary cause and points to a **QKV weight packing/layout issue in the decode path**, likely context‑length dependent.

---

## Metodología / Method

**Configuración común / Common configuration**
- Modelo: `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`
- Flags:
  - `GRETA_TRACE_ATTN_L0_PIPE=1`
  - `GRETA_TRACE_ATTN_L0_NORM=1`
  - `GRETA_TRACE_STAGE_DEBUG_INPUT=1`
  - `GRETA_TRACE_PROMPT_ID=<id>`
- Prompts:
  - `p0_short` ("Hi")
  - `p4_sys` (system template)
  - `p5_ba` (Buenos Aires)
  - `p6_long` (>=200 tokens)

**Comandos clave / Key commands**
- Runner local: `tools/benchmarks/run_b3_32_mi300x.sh 129.212.184.200 2026-02-03`
- Analyzer local: `tools/benchmarks/analyze_b3_32_normout_vs_q.py --dir artifacts_remote/2026-02-03/b3_32/root/gretacore/artifacts/alignment/2026-02-03 --out artifacts_remote/2026-02-03/b3_32/b3_32_analysis.txt`

---

## Resultados / Results

**Tabla principal (prefill_last vs decode0)**

| Prompt | norm_out MAE | q MAE | first mismatch |
|---|---:|---:|---|
| p0_short | 0.0 | 0.0 | NONE |
| p4_sys   | 0.0 | 0.5375176 | q |
| p5_ba    | 0.0 | 0.8888819 | q |
| p6_long  | 0.0 | 0.5396322 | q |

**Interpretación / Interpretation**
- `attn_norm_out` **no** es la causa (MAE=0.0 en todos los prompts).
- La divergencia aparece **en `q`** para prompts con contexto, no en el prompt corto.
- Esto sugiere **mismatch en pesos Q (layout/packing/dequant)** en el path de decode, posiblemente condicionado por contexto / secuencia.

---

## Short vs Long Prompt Behavior (ES)
El prompt corto (`p0_short`) no muestra divergencia (q_mae=0.0), mientras que los prompts con contexto (p4_sys/p5_ba/p6_long) presentan q_mae alto. Esto indica que el problema **no depende del norm_out**, sino del **path de proyección Q en decode** cuando hay contexto acumulado.

## Short vs Long Prompt Behavior (EN)
The short prompt (`p0_short`) shows no divergence (q_mae=0.0), while context prompts (p4_sys/p5_ba/p6_long) show high q_mae. This indicates the issue **does not originate in norm_out**, but in the **decode Q projection path** when context is present.

---

## Evidencia / Evidence
- Artifact TGZ: `artifacts_remote/2026-02-03/b3_32/gretacore_b3_32_artifacts.tgz`
- JSONL: `artifacts_remote/2026-02-03/b3_32/root/gretacore/artifacts/alignment/2026-02-03/b3_32_attn_l0_pipe_*.jsonl`
- Analyzer: `artifacts_remote/2026-02-03/b3_32/b3_32_analysis.txt`

---

## Conclusión y Próximo Paso / Conclusion & Next Step
**Conclusión:** La fuente más probable del mismatch es **QKV weight layout/packing/dequant en decode**, no el input norm_out. 

**Próximo paso (B3.33):** implementar una verificación puntual de layout/packing de pesos Q (y opcionalmente K/V) en decode, con MAE row/col y hash de ventana pequeña, para aislar el layout correcto y corregir la ruta de proyección.

---

**Firma / Signature**

L.E.T / Leandro Emanuel Timberini
