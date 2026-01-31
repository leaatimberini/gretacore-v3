#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer XBuf { float x[]; } xb;
layout(set = 0, binding = 1) buffer YBuf { float y[]; } yb;

layout(push_constant) uniform Push {
  uint rows;
  uint cols;
} pc;

shared float s_max[256];
shared float s_sum[256];

void main() {
  uint row = gl_WorkGroupID.x;
  uint tid = gl_LocalInvocationID.x;
  if (row >= pc.rows)
    return;
  uint base = row * pc.cols;

  float maxv = -3.4e38;
  for (uint c = tid; c < pc.cols; c += 256) {
    float v = xb.x[base + c];
    if (v > maxv)
      maxv = v;
  }
  s_max[tid] = maxv;
  barrier();

  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride)
      s_max[tid] = max(s_max[tid], s_max[tid + stride]);
    barrier();
  }
  float row_max = s_max[0];

  float sum = 0.0;
  for (uint c = tid; c < pc.cols; c += 256) {
    float e = exp(xb.x[base + c] - row_max);
    yb.y[base + c] = e;
    sum += e;
  }
  s_sum[tid] = sum;
  barrier();

  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride)
      s_sum[tid] += s_sum[tid + stride];
    barrier();
  }
  float inv = 1.0 / s_sum[0];

  for (uint c = tid; c < pc.cols; c += 256) {
    yb.y[base + c] = yb.y[base + c] * inv;
  }
}
