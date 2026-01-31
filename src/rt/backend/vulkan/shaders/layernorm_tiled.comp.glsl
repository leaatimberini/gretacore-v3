#version 450

// One workgroup per row; reduce across columns with shared memory.
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer XBuf { float x[]; } xb;
layout(set = 0, binding = 1) buffer GammaBuf { float gamma[]; } gb;
layout(set = 0, binding = 2) buffer BetaBuf { float beta[]; } bb;
layout(set = 0, binding = 3) buffer YBuf { float y[]; } yb;

layout(push_constant) uniform Push {
  uint rows;
  uint cols;
  float eps;
} pc;

shared float s_sum[256];
shared float s_var[256];

void main() {
  uint row = gl_WorkGroupID.x;
  uint tid = gl_LocalInvocationID.x;
  if (row >= pc.rows)
    return;
  uint base = row * pc.cols;

  float sum = 0.0;
  for (uint c = tid; c < pc.cols; c += 256) {
    sum += xb.x[base + c];
  }
  s_sum[tid] = sum;
  barrier();

  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride)
      s_sum[tid] += s_sum[tid + stride];
    barrier();
  }
  float mean = s_sum[0] / float(pc.cols);

  float var = 0.0;
  for (uint c = tid; c < pc.cols; c += 256) {
    float d = xb.x[base + c] - mean;
    var += d * d;
  }
  s_var[tid] = var;
  barrier();

  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride)
      s_var[tid] += s_var[tid + stride];
    barrier();
  }
  float inv = inversesqrt(s_var[0] / float(pc.cols) + pc.eps);

  for (uint c = tid; c < pc.cols; c += 256) {
    float v = (xb.x[base + c] - mean) * inv;
    yb.y[base + c] = v * gb.gamma[c] + bb.beta[c];
  }
}
