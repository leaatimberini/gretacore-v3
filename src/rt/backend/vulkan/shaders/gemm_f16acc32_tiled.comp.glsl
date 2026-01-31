#version 450

// Habilitar FP16 en Vulkan GLSL
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// A y B en FP16 (storage), C en FP32
layout(set = 0, binding = 0) readonly buffer BufA { float16_t a[]; };
layout(set = 0, binding = 1) readonly buffer BufB { float16_t b[]; };
layout(set = 0, binding = 2) writeonly buffer BufC { float c[]; };

layout(push_constant) uniform Push {
    uint M;
    uint N;
    uint K;
    uint lda;
    uint ldb;
    uint ldc;
} pc;

shared float As[16][16];
shared float Bs[16][16];

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;

    uint lx = gl_LocalInvocationID.x;
    uint ly = gl_LocalInvocationID.y;

    bool in_bounds = (row < pc.M) && (col < pc.N);

    float acc = 0.0;

    for (uint k0 = 0; k0 < pc.K; k0 += 16) {
        uint ak = k0 + lx;
        As[ly][lx] = (row < pc.M && ak < pc.K) ? float(a[row * pc.lda + ak]) : 0.0;

        uint bk = k0 + ly;
        Bs[ly][lx] = (bk < pc.K && col < pc.N) ? float(b[bk * pc.ldb + col]) : 0.0;

        barrier();

        for (uint k = 0; k < 16; k++) {
            acc += As[ly][k] * Bs[k][lx];
        }

        barrier();
    }

    if (in_bounds) {
        c[row * pc.ldc + col] = acc;
    }
}
