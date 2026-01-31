#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

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

shared float As[2][16][16];
shared float Bs[2][16][32];

void load_tile(uint buf, uint k0, uint row, uint col0, uint col1, uint lx, uint ly) {
    // A
    uint ak = k0 + lx;
    As[buf][ly][lx] = (row < pc.M && ak < pc.K) ? float(a[row * pc.lda + ak]) : 0.0;

    // B: dos columnas por hilo, bk = k0 + ly
    uint bk = k0 + ly;
    float b0 = 0.0;
    float b1 = 0.0;
    if (bk < pc.K) {
        if (col0 < pc.N) b0 = float(b[bk * pc.ldb + col0]);
        if (col1 < pc.N) b1 = float(b[bk * pc.ldb + col1]);
    }
    Bs[buf][ly][lx * 2u + 0u] = b0;
    Bs[buf][ly][lx * 2u + 1u] = b1;
}

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col0 = gl_GlobalInvocationID.x * 2u;
    uint col1 = col0 + 1u;

    uint lx = gl_LocalInvocationID.x;
    uint ly = gl_LocalInvocationID.y;

    bool in_bounds0 = (row < pc.M) && (col0 < pc.N);
    bool in_bounds1 = (row < pc.M) && (col1 < pc.N);

    float acc0 = 0.0;
    float acc1 = 0.0;

    // Prefetch tile 0
    uint k0 = 0u;
    load_tile(0u, k0, row, col0, col1, lx, ly);
    barrier();

    // Main loop (ping/pong)
    for (; k0 < pc.K; k0 += 16u) {
        uint cur = (k0 / 16u) & 1u;
        uint nxt = cur ^ 1u;

        // Preload next tile while computing current tile (conceptual overlap)
        // Nota: Vulkan no garantiza overlap real, pero reduce estructura de barreras.
        uint k1 = k0 + 16u;
        if (k1 < pc.K) {
            load_tile(nxt, k1, row, col0, col1, lx, ly);
        } else {
            load_tile(nxt, k1, row, col0, col1, lx, ly);
        }

        // Compute current tile
        for (uint k = 0u; k < 16u; k++) {
            float av = As[cur][ly][k];
            acc0 += av * Bs[cur][k][lx * 2u + 0u];
            acc1 += av * Bs[cur][k][lx * 2u + 1u];
        }

        barrier();
        // después del barrier, el nxt tile ya está listo (o fue cargado con ceros si k1>=K)
    }

    if (in_bounds0) c[row * pc.ldc + col0] = acc0;
    if (in_bounds1) c[row * pc.ldc + col1] = acc1;
}
