#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <bit>

namespace gcore::rt::ref {

/**
 * @brief Operaciones de referencia en CPU para validación de correctitud.
 * Estas implementaciones priorizan la claridad y la precisión sobre el rendimiento.
 */
class CpuReference {
public:
    // --- Helpers de Precisión ---

    static uint16_t float_to_half(float f) {
        uint32_t x = std::bit_cast<uint32_t>(f);
        uint32_t sign = (x >> 31) & 0x1;
        int exp = int((x >> 23) & 0xFF) - 127;
        uint32_t mant = x & 0x7FFFFF;
        if (exp > 15) return static_cast<uint16_t>((sign << 15) | 0x7C00);
        if (exp < -14) {
            if (exp < -24) return static_cast<uint16_t>(sign << 15);
            mant |= 0x800000;
            int shift = -exp - 14;
            uint32_t m = mant >> (shift + 13);
            return static_cast<uint16_t>((sign << 15) | m);
        }
        uint32_t exp_h = static_cast<uint32_t>(exp + 15);
        uint32_t mant_h = mant >> 13;
        uint32_t round = mant & 0x1FFF;
        if (round > 0x1000 || (round == 0x1000 && (mant_h & 0x1))) mant_h++;
        if (mant_h == 0x400) { mant_h = 0; exp_h++; if (exp_h >= 31) return static_cast<uint16_t>((sign << 15) | 0x7C00); }
        return static_cast<uint16_t>((sign << 15) | (exp_h << 10) | mant_h);
    }

    static float half_to_float(uint16_t h) {
        uint32_t sign = (h >> 15) & 0x1;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        uint32_t out;
        if (exp == 0) {
            if (mant == 0) out = sign << 31;
            else { exp = 127 - 14; while ((mant & 0x400) == 0) { mant <<= 1; exp--; } mant &= 0x3FF; out = (sign << 31) | (exp << 23) | (mant << 13); }
        } else if (exp == 31) out = (sign << 31) | 0x7F800000 | (mant << 13);
        else { exp = exp + (127 - 15); out = (sign << 31) | (exp << 23) | (mant << 13); }
        return std::bit_cast<float>(out);
    }

    // --- Kernels de Referencia ---

    /**
     * @brief GEMM FP32 de referencia (C = A * B)
     */
    static void gemm(const float* A, const float* B, float* C, int M, int N, int K) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                double acc = 0.0;
                for (int k = 0; k < K; ++k) {
                    acc += static_cast<double>(A[i * K + k]) * static_cast<double>(B[k * N + j]);
                }
                C[i * N + j] = static_cast<float>(acc);
            }
        }
    }

    /**
     * @brief RMSNorm de referencia
     */
    static void rmsnorm(const float* x, float* y, const float* weight, int rows, int cols, float eps = 1e-5f) {
        for (int r = 0; r < rows; r++) {
            const float* xr = x + r * cols;
            float* yr = y + r * cols;
            double ms = 0.0;
            for (int c = 0; c < cols; c++) {
                double v = static_cast<double>(xr[c]);
                ms += v * v;
            }
            ms /= static_cast<double>(cols);
            double inv = 1.0 / std::sqrt(ms + static_cast<double>(eps));
            for (int c = 0; c < cols; c++) {
                yr[c] = static_cast<float>(static_cast<double>(xr[c]) * inv) * weight[c];
            }
        }
    }

    /**
     * @brief LayerNorm de referencia
     */
    static void layernorm(const float* x, float* y, const float* weight, const float* bias, int rows, int cols, float eps = 1e-5f) {
        for (int r = 0; r < rows; r++) {
            const float* xr = x + r * cols;
            float* yr = y + r * cols;
            double mean = 0.0;
            for (int c = 0; c < cols; c++) mean += static_cast<double>(xr[c]);
            mean /= static_cast<double>(cols);
            double var = 0.0;
            for (int c = 0; c < cols; c++) {
                double d = static_cast<double>(xr[c]) - mean;
                var += d * d;
            }
            var /= static_cast<double>(cols);
            double inv = 1.0 / std::sqrt(var + static_cast<double>(eps));
            for (int c = 0; c < cols; c++) {
                yr[c] = static_cast<float>((static_cast<double>(xr[c]) - mean) * inv) * weight[c] + bias[c];
            }
        }
    }

    /**
     * @brief Softmax de referencia
     */
    static void softmax(const float* x, float* y, int rows, int cols) {
        for (int r = 0; r < rows; r++) {
            const float* xr = x + r * cols;
            float* yr = y + r * cols;
            double maxv = static_cast<double>(xr[0]);
            for (int c = 1; c < cols; c++) maxv = std::max(maxv, static_cast<double>(xr[c]));
            double sum = 0.0;
            for (int c = 0; c < cols; c++) {
                double e = std::exp(static_cast<double>(xr[c]) - maxv);
                yr[c] = static_cast<float>(e);
                sum += e;
            }
            double inv = 1.0 / sum;
            for (int c = 0; c < cols; c++) {
                yr[c] = static_cast<float>(static_cast<double>(yr[c]) * inv);
            }
        }
    }
};

} // namespace gcore::rt::ref
