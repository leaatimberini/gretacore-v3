# GRETA CORE - Checklist de Reproducibilidad (Fase 3)

Este checklist asegura que la evidencia técnica de la Fase 3 se recopile y verifique correctamente.

## 1. Verificación del Entorno
- [ ] La versión de ROCm es 6.2 o 7.1.
- [ ] La GPU es detectada como `gfx942` (MI300X).
- [ ] `hipcc --version` devuelve la información esperada del compilador.

## 2. Verificación de Compilación
- [ ] `cmake -DCMAKE_BUILD_TYPE=Release ..` finaliza sin errores.
- [ ] `make -j` produce el binario `greta_infer`.
- [ ] Los metadatos del sistema aparecen al ejecutar con `GRETA_VERBOSE_INFO=1`.

## 3. Verificación de Corrección
- [ ] El prompt "Hola" genera 32 tokens válidos.
- [ ] No hay NaNs/Infs presentes en la salida.
- [ ] (Opcional) `GRETA_TRACE_LEVEL=1` confirma la estabilidad en todas las capas.

## 4. Verificación de Rendimiento
- [ ] Los Tokens/segundo igualan o superan los 12.5 T/s en MI300X.
- [ ] El TTFT es inferior a 250ms para prompts cortos.
- [ ] La ejecución de `./tools/phase3/run_bench.sh` produce un archivo `bench_results.csv` válido.

## 5. Metadatos y Registro
- [ ] Hash del documento registrado para auditoría legal/IP.
- [ ] Modelo GGUF de referencia: `llama-2-7b-chat.Q4_K_M.gguf`.
