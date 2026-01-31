#version 450

// Subgroup (wave) ops
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_shuffle : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

// Kernel layout: one subgroup computes 1 row x 8 cols.
// We rely on pipeline requiredSubgroupSize=32 (VK_EXT_subgroup_size_control).
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

// Buffers:
// A_packed: MxK FP16 packed as half2 in uint32.
// B_packed: KxN FP16 packed as half2 in uint32.
// C:        MxN FP32.
layout(set = 0, binding = 0) readonly buffer BufA { uint  A[]; } bufA;
layout(set = 0, binding = 1) readonly buffer BufB { uint  B[]; } bufB;
layout(set = 0, binding = 2) writeonly buffer BufC { float C[]; } bufC;

// Debug: write gl_SubgroupSize here once per dispatch grid (binding=3).
layout(set = 0, binding = 3) writeonly buffer BufDbg { uint subgroup_size; } dbg;

layout(push_constant) uniform Push {
  uint M;
  uint N;
  uint K;
  uint lda; // leading dim for A (K)
  uint ldb; // leading dim for B (N)
  uint ldc; // leading dim for C (N)
} pc;

float load_fp16_from_half2_u32(const uint packed, const bool hi) {
  vec2 v = unpackHalf2x16(packed);
  return hi ? v.y : v.x;
}

float loadA(uint row, uint k) {
  uint idx_half2 = row * (pc.lda / 2u) + (k >> 1);
  uint packed = bufA.A[idx_half2];
  bool hi = (k & 1u) != 0u;
  return load_fp16_from_half2_u32(packed, hi);
}

float loadB(uint k, uint col) {
  uint idx_half2 = k * (pc.ldb / 2u) + (col >> 1);
  uint packed = bufB.B[idx_half2];
  bool hi = (col & 1u) != 0u;
  return load_fp16_from_half2_u32(packed, hi);
}

void main() {
  // Write subgroup size once (workgroup 0,0 and lane 0)
  if (gl_WorkGroupID.x == 0u && gl_WorkGroupID.y == 0u && gl_SubgroupInvocationID == 0u) {
    dbg.subgroup_size = gl_SubgroupSize;
  }

  uint lane = gl_SubgroupInvocationID;

  uint col_group = lane >> 2;  // 0..7
  uint part      = lane & 3u;  // 0..3

  uint row  = gl_WorkGroupID.y;
  uint col0 = gl_WorkGroupID.x * 8u;

  uint col = col0 + col_group;
  if (row >= pc.M || col >= pc.N) return;

  float acc = 0.0;

  for (uint k = part; k < pc.K; k += 4u) {
    float a = loadA(row, k);
    float b = loadB(k, col);
    acc = fma(a, b, acc);
  }

  float acc1 = subgroupShuffleXor(acc, 1u);
  acc += acc1;
  float acc2 = subgroupShuffleXor(acc, 2u);
  acc += acc2;

  if (part == 0u) {
    uint out_idx = row * pc.ldc + col;
    bufC.C[out_idx] = acc;
  }
}
