GRETA CORE
Un Stack de Cómputo Software-First para Hardware AMD y Modelos de Lenguaje de Gran Escala (LLMs)
1. Resumen (Abstract)

La rápida expansión de los Modelos de Lenguaje de Gran Escala (LLMs) ha expuesto una debilidad estructural crítica en el ecosistema moderno de inteligencia artificial: una dependencia casi total de la plataforma CUDA de NVIDIA. Esta dependencia ha generado un monopolio de facto sobre el cómputo para IA, impulsando los costos de hardware a niveles insostenibles y limitando artificialmente la innovación.

GRETA CORE es una iniciativa de ingeniería a largo plazo cuyo objetivo es romper esta dependencia mediante la construcción de un stack de cómputo mínimo, de alto rendimiento y de estilo CUDA para hardware AMD, diseñado específicamente para cargas de trabajo de LLMs.

En lugar de competir a nivel de hardware, GRETA CORE se enfoca en la dominación por software: control total del runtime, las bibliotecas de kernels, la gestión de memoria y el modelo de ejecución. El proyecto se construye desde primeros principios, priorizando rendimiento, transparencia y sostenibilidad a largo plazo por encima de la compatibilidad superficial o de corto plazo.

2. Planteamiento del Problema
2.1 El Lock-in de CUDA

CUDA no es simplemente una API de programación; es un ecosistema completo que acopla estrechamente hardware, software, herramientas y flujos de trabajo de desarrollo. Con el tiempo, este acoplamiento ha creado un bucle auto-reforzado:

Los frameworks optimizan primero para CUDA.

Las herramientas asumen semánticas CUDA.

Los desarrolladores eligen hardware NVIDIA por defecto.

Las plataformas alternativas son tratadas como secundarias o experimentales.

Como resultado, el ecosistema de IA ha dejado de ser agnóstico al hardware. Hoy es CUDA-céntrico.

2.2 Aumento del Costo del Hardware

La dominancia de CUDA ha generado una escasez artificial de hardware “utilizable” para IA. GPUs que son técnicamente capaces de ejecutar cargas de trabajo de LLMs quedan excluidas por limitaciones de software, no por restricciones físicas reales.

Esto ha provocado:

Incrementos exponenciales en el precio de GPUs.

Menor accesibilidad para desarrolladores independientes e investigadores.

Centralización de las capacidades de IA en grandes organizaciones.

El problema no es el rendimiento del hardware, sino la disponibilidad y optimización del software.

2.3 Hardware AMD: Capaz pero Subutilizado

AMD produce CPUs, GPUs y APUs competitivos, con arquitecturas modernas y jerarquías de memoria avanzadas. Sin embargo, el hardware AMD está sistemáticamente subutilizado en cargas de trabajo de LLMs debido a:

Stacks de software fragmentados.

Abstracciones incompletas o excesivamente genéricas.

Kernels críticos para el rendimiento que quedan rezagados frente a sus equivalentes en CUDA.

Herramientas que priorizan amplitud de soporte sobre profundidad y especialización.

La ausencia de un stack de cómputo LLM-first ha dejado al hardware AMD operando muy por debajo de su potencial real.

3. Filosofía

GRETA CORE se rige por un conjunto de principios no negociables.

3.1 Software por Sobre Hardware

Las limitaciones de hardware son finitas. Las limitaciones de software no lo son.

GRETA CORE parte de la premisa de que el software es el principal cuello de botella para democratizar el cómputo de IA. Al dominar el stack de software, el hardware existente puede ser llevado mucho más allá de sus usos actuales.

3.2 Control Total del Stack

El rendimiento no se logra mediante wrappers ni capas de abstracción genéricas.

GRETA CORE busca control total sobre:

Ejecución del runtime.

Asignación y reutilización de memoria.

Planificación de kernels.

Movimiento de datos.

Autotuning y fusión de operaciones.

Los componentes externos solo se utilizan si aportan valor medible y pueden ser modificados, adaptados o reemplazados cuando sea necesario.

3.3 Minimalismo y Rendimiento

Cada capa de abstracción introduce overhead.

GRETA CORE rechaza el bloat, la generalidad innecesaria y las dependencias superfluas. El stack es deliberadamente estrecho y está optimizado para una clase específica de cargas de trabajo: inferencia de LLMs y patrones de cómputo asociados.

Si un componente no mejora el rendimiento, la estabilidad o el control del desarrollador, no pertenece al sistema.

4. Qué es GRETA CORE

GRETA CORE es:

Un runtime de cómputo personalizado para hardware AMD.

Un stack de ejecución LLM kernel-first.

Una experiencia de desarrollo tipo CUDA, sin replicar CUDA.

Un esfuerzo de investigación e ingeniería a largo plazo, no un producto inmediato.

Una plataforma diseñada para evolucionar junto con las arquitecturas de LLMs.

5. Qué NO es GRETA CORE

GRETA CORE no es:

Un fork de CUDA.

Un wrapper delgado sobre frameworks existentes.

Una plataforma de cómputo GPU de propósito general.

Un proyecto de optimización de corto plazo.

Un competidor enfocado en paridad de marketing en lugar de sustancia técnica.

La compatibilidad es un objetivo, pero el rendimiento y el control tienen prioridad.

6. Visión Técnica
6.1 Runtime

El runtime de GRETA CORE es responsable de:

Gestión explícita de streams y eventos.

Planificación determinista de kernels.

Pooling y reutilización de memoria de alto rendimiento.

Mecanismos de lanzamiento de kernels de bajo overhead.

Telemetría y profiling integrados.

El runtime está diseñado para minimizar la interacción con el sistema operativo durante la ejecución en estado estable.

6.2 Bibliotecas de Kernels

El núcleo de GRETA CORE reside en sus implementaciones de kernels.

Las áreas iniciales de enfoque incluyen:

GEMM (FP16, BF16 y variantes cuantizadas).

LayerNorm y RMSNorm.

Softmax y primitivas relacionadas con atención.

Operaciones de gestión de KV-cache.

Kernels fusionados para minimizar tráfico de memoria.

La corrección de los kernels es obligatoria.
El rendimiento de los kernels es prioritario.

6.3 Compilador y Autotuning

GRETA CORE no se basa únicamente en kernels estáticos. Incorporará:

Exploración de parámetros de kernels.

Autotuning consciente del hardware.

Fusión controlada de operaciones.

Modelos de costo basados en datos empíricos.

Esto permite adaptarse a distintas arquitecturas AMD sin sacrificar rendimiento.

6.4 Integración con Frameworks

GRETA CORE no busca reemplazar frameworks existentes. En su lugar, se integra de forma selectiva mediante:

Execution providers personalizados.

Bridges de runtime mínimos.

Rutas de invocación directa para cargas de trabajo críticas.

El objetivo es permitir la ejecución de LLMs sin obligar a los desarrolladores a abandonar herramientas conocidas.

7. Hoja de Ruta a Largo Plazo

GRETA CORE es una iniciativa de varios años.

Las fases de alto nivel incluyen:

Runtime fundacional y benchmarking.

Paridad de rendimiento en kernels LLM críticos.

Pipelines completos de inferencia de LLMs.

Herramientas para desarrolladores y profiling avanzado.

Expansión del ecosistema y soporte de modelos más amplios.

Cada fase se evalúa mediante criterios medibles de rendimiento y estabilidad.

8. Impacto en el Ecosistema de IA

Al reducir la barrera de software para el cómputo efectivo en hardware AMD, GRETA CORE busca:

Incrementar la competencia en el mercado de hardware para IA.

Reducir costos para desarrolladores y organizaciones.

Descentralizar el acceso a capacidades de LLMs.

Fomentar innovación más allá de un ecosistema de proveedor único.

Este impacto se logra mediante ingeniería, no mediante regulación.

9. Conclusión

La dominancia actual de CUDA no es inevitable. Es el resultado de una inversión sostenida en software, no de una superioridad técnica insuperable.

GRETA CORE existe para demostrar que el control del software, el minimalismo y la disciplina de ingeniería a largo plazo pueden liberar todo el potencial de plataformas de hardware alternativas.

Este proyecto no es fácil.
No es rápido.
Y no está garantizado.

Pero es necesario.

📌 Estado del Documento

Versión: 1.0

Estado: Borrador Fundacional

Fase del Proyecto: Fase 0 – Fundaciones

Idioma: Español

## Autoría

GRETA CORE es un proyecto de ingeniería independiente concebido,
fundado y liderado por:

Leandro Emanuel Timberini  
Fundador y Arquitecto Principal de Sistemas

Todas las decisiones arquitectónicas, la visión a largo plazo
y los principios fundacionales se originan en esta autoría.
