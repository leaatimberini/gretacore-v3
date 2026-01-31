#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer XBuf { float x[]; } xb;
layout(set = 0, binding = 1) buffer GammaBuf { float gamma[]; } gb;
layout(set = 0, binding = 2) buffer YBuf { float y[]; } yb;

layout(push_constant) uniform Push {
  uint rows;
  uint cols;
  float eps;
} pc;

shared float s_ms[256];

void main() {
  uint row = gl_WorkGroupID.x;
  uint tid = gl_LocalInvocationID.x;
  if (row >= pc.rows)
    return;
  uint base = row * pc.cols;

  float ms = 0.0;
  for (uint c = tid; c < pc.cols; c += 256) {
    float v = xb.x[base + c];
    ms += v * v;
  }
  s_ms[tid] = ms;
  barrier();

  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride)
      s_ms[tid] += s_ms[tid + stride];
    barrier();
  }
  float inv = inversesqrt(s_ms[0] / float(pc.cols) + pc.eps);

  for (uint c = tid; c < pc.cols; c += 256) {
    float v = xb.x[base + c] * inv;
    yb.y[base + c] = v * gb.gamma[c];
  }
}
