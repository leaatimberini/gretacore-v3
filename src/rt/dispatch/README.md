# GRETA CORE — Runtime Dispatch

Path: src/rt/dispatch/README.md  
Version: 1.0  
Language: EN/ES (bilingual)

## EN
Defines the dispatch interface for executing work on a Stream.
In v1, dispatch runs CPU callables but preserves the model used by GPU runtimes:
- enqueue work
- record events
- collect telemetry

## ES
Define la interfaz de dispatch para ejecutar trabajo en un Stream.
En v1, dispatch ejecuta callables CPU pero preserva el modelo de runtimes GPU:
- encolar trabajo
- registrar eventos
- recolectar telemetría
