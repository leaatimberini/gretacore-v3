#version 450

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer XBuf { float x[]; } xb;
layout(set = 0, binding = 1) buffer YBuf { float y[]; } yb;

layout(push_constant) uniform Push {
  uint rows;
  uint cols;
} pc;

void main() {
  uint row = gl_GlobalInvocationID.x;
  if (row >= pc.rows)
    return;
  uint base = row * pc.cols;

  float maxv = xb.x[base + 0];
  for (uint c = 1; c < pc.cols; ++c) {
    float v = xb.x[base + c];
    if (v > maxv)
      maxv = v;
  }

  float sum = 0.0;
  for (uint c = 0; c < pc.cols; ++c) {
    float e = exp(xb.x[base + c] - maxv);
    yb.y[base + c] = e;
    sum += e;
  }
  float inv = 1.0 / sum;
  for (uint c = 0; c < pc.cols; ++c) {
    yb.y[base + c] = yb.y[base + c] * inv;
  }
}
