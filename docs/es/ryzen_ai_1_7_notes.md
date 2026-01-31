# GRETA CORE – Notas Ryzen AI 1.7 (Plan de Extracción)

Versión: 0.1  
Estado: Investigación / Pendiente de Validación  
Fase del Proyecto: Fase 0 – Fundaciones  
Idioma: Español

---

## Propósito de este Documento

Ryzen AI Software 1.7 es una implementación de referencia del stack
runtime+driver actual de AMD para cargas de trabajo de IA en hardware Ryzen AI.

Este documento define **qué debemos extraer, cómo lo vamos a extraer** y cómo
se mapea a la arquitectura de largo plazo de GRETA CORE (runtime + ABI de driver).

No tratamos Ryzen AI como solución final; lo tratamos como
*referencia de realidad* para entender restricciones de drivers y
patrones productivos válidos.

---

## Alcance

Este documento se enfoca en **extracción de información**, no en copiar ni
reimplementar el software de AMD.

Extraemos únicamente:
- Comportamiento del runtime
- Requisitos y restricciones de drivers
- Detalles del pipeline de ejecución de modelos
- Semántica de cache y artefactos
- Patrones de empaquetado y dependencias

---

## Datos a Extraer (Obligatorio)

1) **Componentes del Stack Runtime**
   - Identificar el stack exacto que usa Ryzen AI 1.7 (framework, EPs,
     componentes de compilación/traducción).
   - Confirmar qué componentes son obligatorios vs opcionales.

2) **Dependencias de Driver y Firmware**
   - Versiones mínimas requeridas de driver para NPU/iGPU.
   - Versiones de firmware y mecanismos de actualización (si existen).

3) **Matriz de Hardware Soportado**
   - Generaciones de APU y device IDs soportados.
   - Distinguir disponibilidad de NPU vs iGPU.

4) **Restricciones de Modelo y Formato**
   - Formatos preferidos (ONNX, etc.).
   - Restricciones de opset y tipos (FP16/BF16/INT8).

5) **Comportamiento de Cache**
   - Dónde se guardan artefactos compilados.
   - Estructura de la clave de cache (device/driver/model/opset).
   - Política de persistencia y descarte.

6) **Ruta de Ejecución**
   - Flujo típico: load → compile → cache → execute → telemetry.
   - Directorios temporales y artefactos intermedios.

7) **Entorno y Configuración**
   - Variables de entorno o claves de registro usadas por el runtime.
   - Toggles de debug/telemetría (si existen).

8) **Empaquetado y Deployment**
   - DLLs/SOs requeridos para ejecución.
   - Conjunto mínimo para redistribución.

---

## Procedimiento de Extracción (Planificado)

1) **Instalar Ryzen AI Software 1.7** en una máquina de referencia.
2) Recolectar **logs del instalador** y manifests de versiones.
3) Enumerar dependencias del runtime:
   - Windows: listar DLLs y dependencias.
   - Linux: listar SOs y ejecutar `ldd`.
4) Ejecutar un **modelo de ejemplo oficial** y capturar:
   - Logs
   - Directorios de cache
   - Artefactos del runtime
5) Registrar:
   - Device IDs
   - Versiones de drivers
   - Versiones de componentes del runtime
6) Comparar con GRETA CORE y documentar brechas.

---

## Mapeo a GRETA CORE (Guías)

- **Versionado de ABI de driver**: definir una superficie ABI estable entre
  runtime GRETA y drivers GPU/NPU.
- **Cache determinista**: incluir version de driver + device ID en la clave.
- **Feature gating**: checks explícitos de capacidades (FP16/BF16/INT8).
- **Política de fallback**: camino seguro por defecto para drivers inestables.

---

## Compatibilidad Oficial (Baseline Actual)

- En la página oficial del producto, AMD afirma que el Ryzen 5 8600G incluye
  “AMD Ryzen AI”. Este es nuestro hardware base actual y debe figurar en la
  matriz de compatibilidad de GRETA.
  Fuente: https://www.amd.com/es/products/processors/desktops/ryzen/8000-series/amd-ryzen-5-8600g.html

---

## Preguntas Abiertas (A Resolver)

- ¿Qué backend(s) runtime son obligatorios en Ryzen AI 1.7?
- ¿Cómo se compilan kernels (offline vs JIT)?
- ¿Qué metadata se usa para key de cache?
- ¿Cuál es el paquete mínimo redistribuible?
- ¿Cómo se gatean features entre NPU e iGPU?

---

## Próximas Acciones

- Ejecutar el procedimiento de extracción en un sistema Ryzen AI real.
- Completar este documento con **hechos validados** (versiones, paths,
  dependencias y comportamiento).
- Mantener la versión en inglés sincronizada.
