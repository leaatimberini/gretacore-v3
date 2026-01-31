# GRETA CORE – Ryzen AI 1.7 Notes (Extraction Plan)

Version: 0.1  
Status: Research / Pending Validation  
Project Phase: Phase 0 – Foundations  
Language: English

---

## Purpose of This Document

Ryzen AI Software 1.7 is a reference implementation of AMD’s current
runtime+driver stack for AI workloads on Ryzen AI hardware.

This document defines **what we must extract, how we will extract it**, and
how it maps into GRETA CORE’s long‑term architecture (runtime + driver ABI).

We do **not** treat Ryzen AI as a final solution; we treat it as a
*ground‑truth reference* to understand real driver constraints and
successful production patterns.

---

## Scope

This document focuses on **information extraction**, not on copying or
reimplementing AMD’s software.

We explicitly extract only:
- Runtime behavior
- Driver requirements and constraints
- Model execution pipeline details
- Cache and artifact semantics
- Packaging/runtime dependency patterns

---

## Data to Extract (Required)

1) **Runtime Stack Components**
   - Identify the exact runtime stack used by Ryzen AI 1.7 (framework, EPs,
     compiler/translator components).
   - Confirm which components are mandatory vs optional.

2) **Driver and Firmware Dependencies**
   - Minimal required driver versions for NPU/iGPU.
   - Firmware versions and update mechanisms (if exposed).

3) **Supported Hardware Matrix**
   - APU generations and device IDs supported.
   - Distinguish between NPU vs iGPU availability.

4) **Model and Format Constraints**
   - Preferred model formats (ONNX, etc.).
   - Opset constraints, supported datatypes (FP16/BF16/INT8).

5) **Cache Behavior**
   - Where compiled artifacts are stored.
   - Cache key structure (device/driver/model/opset).
   - Eviction policy and persistence.

6) **Execution Path**
   - Typical flow: load → compile → cache → execute → telemetry.
   - Staging directories or temporary artifacts.

7) **Environment and Configuration**
   - Environment variables or registry keys used by the runtime.
   - Debug/telemetry toggles (if any).

8) **Packaging and Deployment**
   - Runtime DLLs/SOs required for deployment.
   - Minimal set for redistribution.

---

## Extraction Procedure (Planned)

1) **Install Ryzen AI Software 1.7** on a reference machine.
2) Collect **installer logs** and version manifests.
3) Enumerate runtime dependencies:
   - Windows: list installed DLLs and check dependencies.
   - Linux: list shared objects and run `ldd`.
4) Run an **official sample model** and capture:
   - Logs
   - Cache directories
   - Runtime artifacts
5) Record:
   - Device IDs
   - Driver versions
   - Runtime component versions
6) Compare with GRETA CORE architecture and document gaps.

---

## Mapping to GRETA CORE (Guidelines)

- **Driver ABI versioning**: define a stable, explicit ABI surface between
  GRETA runtime and GPU/NPU drivers.
- **Cache determinism**: include driver version + device ID in GRETA cache key.
- **Feature gating**: enforce explicit capability checks (FP16/BF16/INT8).
- **Fallback policy**: safe default path for unstable drivers (GPU-safe mode).

---

## Official Compatibility (Current Baseline)

- AMD’s official product page for the Ryzen 5 8600G explicitly states that it
  includes “AMD Ryzen AI”. This is our current hardware baseline and should be
  tracked in the GRETA compatibility matrix.
  Source: https://www.amd.com/es/products/processors/desktops/ryzen/8000-series/amd-ryzen-5-8600g.html

---

## Open Questions (To Be Resolved)

- Which exact runtime backend(s) are mandatory in Ryzen AI 1.7?
- How are kernels compiled (offline vs JIT)?
- What metadata is used to key caches?
- What is the minimum viable runtime package for redistribution?
- How does AMD gate features across NPU vs iGPU?

---

## Next Actions

- Perform the extraction procedure on a reference Ryzen AI system.
- Populate this document with **validated facts** (version numbers, paths,
  dependencies, and behavior).
- Mirror the finalized document into the Spanish version.
