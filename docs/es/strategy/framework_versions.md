# Matriz de Versiones de Frameworks (Borrador)

Fecha: 2026-01-31

## Objetivo
Fijar versiones de frameworks incluidas en GRETA para que la instalación sea determinista e idéntica entre Radeon dev y MI300X cloud.

## Targets
- **Radeon Dev** (APU/dGPU)
- **MI300X Cloud** (Instinct)

## Matriz de Versiones (Candidata)
| Componente | Radeon Dev | MI300X Cloud | Notas |
| --- | --- | --- | --- |
| ROCm | 7.1.1 (candidato) | 7.1.1 (candidato) | ROCm 7.1.1 soporta PyTorch 2.9. citeturn1search4 |
| torch | 2.9 (build ROCm) | 2.9 (build ROCm) | ROCm 7.1.1 habilita soporte PyTorch 2.9. citeturn1search4 |
| triton | Incluido en wheels PyTorch ROCm | Incluido en wheels PyTorch ROCm | AMD instala Triton vía PyTorch ROCm. citeturn1search2 |
| jax | TBD (bloqueado) | TBD (bloqueado) | Path ROCm para JAX inconsistente; no hay extra oficial. citeturn0search3 |

## Lock Files
- `tools/compat/lock/greta_requirements.txt`
- `tools/compat/lock/greta_environment.yml`

## Criterio de Pase
- Mismas versiones entre targets (salvo requisitos del vendor).
- Prototipos de compatibilidad con `STATUS=OK`.
- Path de instalación JAX ROCm resuelto y pinneado.
