#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

// Workgroup: 16x8 = 128 threads
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;

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

// Tile: 8 filas x 16 K (A), B: 16 K x 32 cols
shared float As[8][16];
shared float Bs[16][32];

void main() {
    uint row = gl_WorkGroupID.y * 8u + gl_LocalInvocationID.y; // 8 filas por WG
    uint col0 = (gl_WorkGroupID.x * 16u + gl_LocalInvocationID.x) * 2u; // 2 cols por hilo
    uint col1 = col0 + 1u;

    uint lx = gl_LocalInvocationID.x; // 0..15
    uint ly = gl_LocalInvocationID.y; // 0..7

    bool in_bounds0 = (row < pc.M) && (col0 < pc.N);
    bool in_bounds1 = (row < pc.M) && (col1 < pc.N);

    float acc0 = 0.0;
    float acc1 = 0.0;

    for (uint k0 = 0; k0 < pc.K; k0 += 16u) {
        // A: cada hilo carga 1 elemento (8x16 = 128)
        uint ak = k0 + lx;
        As[ly][lx] = (row < pc.M && ak < pc.K) ? float(a[row * pc.lda + ak]) : 0.0;

        // B: necesitamos llenar Bs[0..15][0..31]
        // Con 128 hilos: cada hilo puede cargar 2 elementos totales
        // Mapeo: cada hilo carga B para su (bk = k0+ly) y sus 2 cols, pero eso sólo llena 8 filas de B.
        // Para completar las 16 filas de B, usamos una segunda carga para bk = k0 + (ly+8) si aplica.
        uint bk0 = k0 + ly;
        float b00 = 0.0, b01 = 0.0;
        if (bk0 < pc.K) {
            if (col0 < pc.N) b00 = float(b[bk0 * pc.ldb + col0]);
            if (col1 < pc.N) b01 = float(b[bk0 * pc.ldb + col1]);
        }
        Bs[ly][lx * 2u + 0u] = b00;
        Bs[ly][lx * 2u + 1u] = b01;

        uint bk1 = k0 + (ly + 8u);
        float b10 = 0.0, b11 = 0.0;
        if (bk1 < pc.K) {
            if (col0 < pc.N) b10 = float(b[bk1 * pc.ldb + col0]);
            if (col1 < pc.N) b11 = float(b[bk1 * pc.ldb + col1]);
        }
        Bs[ly + 8u][lx * 2u + 0u] = b10;
        Bs[ly + 8u][lx * 2u + 1u] = b11;

        barrier();

        for (uint k = 0; k < 16u; k++) {
            float av = As[ly][k];
            acc0 += av * Bs[k][lx * 2u + 0u];
            acc1 += av * Bs[k][lx * 2u + 1u];
        }

        barrier();
    }

    if (in_bounds0) c[row * pc.ldc + col0] = acc0;
    if (in_bounds1) c[row * pc.ldc + col1] = acc1;
}
