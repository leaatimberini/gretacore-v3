#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

// Cada hilo calcula 2 columnas (col y col+1)
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// A y B en FP16, C en FP32
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

// Shared: A 16x16 (fp32) y B 16x(16*2) porque guardamos 2 cols por hilo
shared float As[16][16];
shared float Bs[16][32];

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col0 = gl_GlobalInvocationID.x * 2u; // 2 cols por hilo
    uint col1 = col0 + 1u;

    uint lx = gl_LocalInvocationID.x; // 0..15
    uint ly = gl_LocalInvocationID.y; // 0..15

    bool in_bounds0 = (row < pc.M) && (col0 < pc.N);
    bool in_bounds1 = (row < pc.M) && (col1 < pc.N);

    float acc0 = 0.0;
    float acc1 = 0.0;

    for (uint k0 = 0; k0 < pc.K; k0 += 16u) {
        // Load A tile: cada hilo carga 1 elemento
        uint ak = k0 + lx;
        As[ly][lx] = (row < pc.M && ak < pc.K) ? float(a[row * pc.lda + ak]) : 0.0;

        // Load B tile: cada hilo carga 2 columnas (col0/col1) para 1 k (bk)
        uint bk = k0 + ly;
        float b0 = 0.0;
        float b1 = 0.0;
        if (bk < pc.K) {
            if (col0 < pc.N) b0 = float(b[bk * pc.ldb + col0]);
            if (col1 < pc.N) b1 = float(b[bk * pc.ldb + col1]);
        }
        Bs[ly][lx * 2u + 0u] = b0;
        Bs[ly][lx * 2u + 1u] = b1;

        barrier();

        // Compute
        for (uint k = 0; k < 16u; k++) {
            float av = As[ly][k];
            acc0 += av * Bs[k][lx * 2u + 0u];
            acc1 += av * Bs[k][lx * 2u + 1u];
        }

        barrier();
    }

    // Store
    if (in_bounds0) c[row * pc.ldc + col0] = acc0;
    if (in_bounds1) c[row * pc.ldc + col1] = acc1;
}
