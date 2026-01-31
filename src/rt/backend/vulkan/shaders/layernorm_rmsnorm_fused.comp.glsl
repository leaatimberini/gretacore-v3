#version 450

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer XBuf { float x[]; } xb;
layout(set = 0, binding = 1) buffer GammaLnBuf { float gamma_ln[]; } glb;
layout(set = 0, binding = 2) buffer BetaLnBuf { float beta_ln[]; } blb;
layout(set = 0, binding = 3) buffer GammaRmsBuf { float gamma_rms[]; } grb;
layout(set = 0, binding = 4) buffer YLnBuf { float y_ln[]; } yln;
layout(set = 0, binding = 5) buffer YRmsBuf { float y_rms[]; } yrm;

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
  float ms = 0.0;
  for (uint c = 0; c < pc.cols; ++c) {
    float v = xb.x[base + c];
    mean += v;
    ms += v * v;
  }
  mean /= float(pc.cols);
  ms /= float(pc.cols);
  float var = ms - mean * mean;
  if (var < 0.0)
    var = 0.0;

  float inv_ln = inversesqrt(var + pc.eps);
  float inv_rms = inversesqrt(ms + pc.eps);

  for (uint c = 0; c < pc.cols; ++c) {
    float v = xb.x[base + c];
    float ln = (v - mean) * inv_ln;
    yln.y_ln[base + c] = ln * glb.gamma_ln[c] + blb.beta_ln[c];
    yrm.y_rms[base + c] = v * inv_rms * grb.gamma_rms[c];
  }
}
