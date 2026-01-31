# GRETA CORE — Runtime Allocator

Path: src/rt/allocator/README.md  
Version: 1.0  
Language: EN/ES (bilingual)

## EN — Goal
Provide a minimal, high-performance host allocator for runtime workloads:
- fast alloc/free
- memory reuse via pooling
- predictable behavior
- low overhead

Strategy:
- power-of-two size classes (bins)
- freelists per bin
- large allocations bypass bins (direct allocation)

## ES — Objetivo
Proveer un allocator de host mínimo y de alto rendimiento para el runtime:
- alloc/free rápidos
- reutilización de memoria mediante pooling
- comportamiento predecible
- bajo overhead

Estrategia:
- clases de tamaño por potencias de 2 (bins)
- freelists por bin
- allocations grandes bypass (asignación directa)
