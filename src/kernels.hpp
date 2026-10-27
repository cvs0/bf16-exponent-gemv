#pragma once

#include "expcode/gemv.hpp"

namespace expcode {

void gemv_encoded_avx2_range(const MatrixView& m, const float* x, float* y, int r0, int r1);
void gemv_encoded_avx512_range(const MatrixView& m, const float* x, float* y, int r0, int r1);
void gemv_bf16_avx2_range(const uint16_t* W, const float* x, float* y, int cols, int r0, int r1);
void gemv_bf16_avx512_range(const uint16_t* W, const float* x, float* y, int cols, int r0, int r1);
void gemv_f32_avx2_range(const float* W, const float* x, float* y, int cols, int r0, int r1);
void gemv_f32_avx512_range(const float* W, const float* x, float* y, int cols, int r0, int r1);

double peak_fma_gflops_avx2();
double peak_fma_gflops_avx512();

float stream_read_f32_avx512(const float* a, std::size_t n, int threads);
void stream_triad_f32_avx512(float* a, const float* b, const float* c, std::size_t n, float scalar,
                             int threads, bool nontemporal);
void stream_copy_f32_avx512(float* dst, const float* src, std::size_t n, int threads);

} // namespace expcode
