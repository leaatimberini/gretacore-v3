#version 450

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer XBuf { float x[]; } xb;
layout(set = 0, binding = 1) buffer GammaBuf { float gamma[]; } gb;
layout(set = 0, binding = 2) buffer BetaBuf { float beta[]; } bb;
layout(set = 0, binding = 3) buffer YBuf { float y[]; } yb;

layout(push_constant) uniform Push {
  uint rows;
  uint cols;
  float eps;
} pc;

void main() {
  uint row = gl_GlobalInvocationID.x;
  if (row >= pc.rows)
    return;
  uint base = row * pc.cols;

  float mean = 0.0;
  for (uint c = 0; c < pc.cols; ++c) {
    mean += xb.x[base + c];
  }
  mean /= float(pc.cols);

  float var = 0.0;
  for (uint c = 0; c < pc.cols; ++c) {
    float d = xb.x[base + c] - mean;
    var += d * d;
  }
  var /= float(pc.cols);
  float inv = inversesqrt(var + pc.eps);

  for (uint c = 0; c < pc.cols; ++c) {
    float v = (xb.x[base + c] - mean) * inv;
    yb.y[base + c] = v * gb.gamma[c] + bb.beta[c];
  }
}
