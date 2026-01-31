#version 450

// One workgroup per row; reduce across columns with shared memory.
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

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

shared float s_sum[256];
shared float s_sumsq[256];

void main() {
  uint row = gl_WorkGroupID.x;
  uint tid = gl_LocalInvocationID.x;
  if (row >= pc.rows)
    return;
  uint base = row * pc.cols;

  float sum = 0.0;
  float sumsq = 0.0;
  for (uint c = tid; c < pc.cols; c += 256) {
    float v = xb.x[base + c];
    sum += v;
    sumsq += v * v;
  }
  s_sum[tid] = sum;
  s_sumsq[tid] = sumsq;
  barrier();

  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
      s_sumsq[tid] += s_sumsq[tid + stride];
    }
    barrier();
  }

  float mean = s_sum[0] / float(pc.cols);
  float ms = s_sumsq[0] / float(pc.cols);
  float var = ms - mean * mean;
  if (var < 0.0)
    var = 0.0;

  float inv_ln = inversesqrt(var + pc.eps);
  float inv_rms = inversesqrt(ms + pc.eps);

  for (uint c = tid; c < pc.cols; c += 256) {
    float v = xb.x[base + c];
    float ln = (v - mean) * inv_ln;
    yln.y_ln[base + c] = ln * glb.gamma_ln[c] + blb.beta_ln[c];
    yrm.y_rms[base + c] = v * inv_rms * grb.gamma_rms[c];
  }
}
