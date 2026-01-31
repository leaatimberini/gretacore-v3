#version 450

// 16x16 = 256 invocations/workgroup (safe default)
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer BufA { float a[]; };
layout(set = 0, binding = 1) readonly buffer BufB { float b[]; };
layout(set = 0, binding = 2) writeonly buffer BufC { float c[]; };

layout(push_constant) uniform Push {
    uint M;
    uint N;
    uint K;
    uint lda;   // leading dimension A (K if row-major)
    uint ldb;   // leading dimension B (N if row-major)
    uint ldc;   // leading dimension C (N if row-major)
} pc;

// Row-major indexing
// A: MxK, index = row*lda + k
// B: KxN, index = k*ldb + col
// C: MxN, index = row*ldc + col
void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;

    if (row >= pc.M || col >= pc.N) return;

    float acc = 0.0;
    uint a_base = row * pc.lda;
    for (uint k = 0; k < pc.K; k++) {
        float av = a[a_base + k];
        float bv = b[k * pc.ldb + col];
        acc += av * bv;
    }

    c[row * pc.ldc + col] = acc;
}
