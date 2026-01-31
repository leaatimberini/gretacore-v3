# GRETA CORE — Vulkan Shaders

Path: src/rt/backend/vulkan/shaders/README.md  
Version: 1.0

We store minimal GLSL compute shaders here.
Bench builds compile GLSL -> SPIR-V via glslangValidator.

Initial shader:
- fill.comp.glsl: write a constant uint value into an SSBO.
