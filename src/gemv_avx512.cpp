#include "kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

#include <immintrin.h>
#include <omp.h>

namespace expcode {
namespace {

inline bool block_has_escape(const uint8_t* codes) {
    const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes));
    const __m256i nibble = _mm256_set1_epi8(0x0F);
    const __m256i lo = _mm256_and_si256(raw, nibble);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), nibble);
    const __m256i esc = _mm256_set1_epi8(char(kEscapeCode));
    const __m256i hit = _mm256_or_si256(_mm256_cmpeq_epi8(lo, esc), _mm256_cmpeq_epi8(hi, esc));
    return _mm256_testz_si256(hit, hit) == 0;
}

inline __m512 decode16_ps(const uint8_t* codebook, const uint8_t* code_bytes, const uint8_t* sm) {
    const __m128i table = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codebook));
    const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(code_bytes));
    const __m128i nibble = _mm_set1_epi8(0x0F);
    const __m128i lo = _mm_and_si128(raw, nibble);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), nibble);
    const __m128i exps =
        _mm_unpacklo_epi8(_mm_shuffle_epi8(table, lo), _mm_shuffle_epi8(table, hi));
    const __m128i smv = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sm));
    const __m512i sm32 = _mm512_cvtepu8_epi32(smv);
    const __m512i e32 = _mm512_cvtepu8_epi32(exps);
    const __m512i sign = _mm512_slli_epi32(_mm512_and_si512(sm32, _mm512_set1_epi32(0x80)), 24);
    const __m512i mant = _mm512_slli_epi32(_mm512_and_si512(sm32, _mm512_set1_epi32(0x7F)), 16);
    const __m512i eb = _mm512_slli_epi32(e32, 23);
    return _mm512_castsi512_ps(_mm512_or_si512(sign, _mm512_or_si512(mant, eb)));
}

inline __m512 bf16_16_to_ps(const uint16_t* p) {
    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(v), 16));
}

inline void store_acc4(__m512 acc[4], float partial[kAccStripe]) {
    for (int k = 0; k < 4; ++k)
        _mm512_storeu_ps(partial + 16 * k, acc[k]);
}

inline void load_acc4(const float partial[kAccStripe], __m512 acc[4]) {
    for (int k = 0; k < 4; ++k)
        acc[k] = _mm512_loadu_ps(partial + 16 * k);
}

} // namespace

void gemv_encoded_avx512_range(const MatrixView& m, const float* x, float* y, int r0, int r1) {
    const int cols = int(m.cols);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* cb = m.codebook_for_row(uint32_t(r));
        const uint8_t* codes = m.codes + std::size_t(r) * m.code_stride;
        const uint8_t* sm = m.sign_mant + std::size_t(r) * m.sm_stride;
        const uint8_t* esc = m.escapes + m.escape_offset_at(uint32_t(r));
        __m512 acc[4];
        for (int k = 0; k < 4; ++k)
            acc[k] = _mm512_setzero_ps();
        int c = 0;
        for (; c + kAccStripe <= cols; c += kAccStripe) {
            if (block_has_escape(codes + (c >> 1))) {
                float partial[kAccStripe];
                store_acc4(acc, partial);
                accum_encoded_range(cb, codes, sm, esc, x, partial, c, c + kAccStripe);
                load_acc4(partial, acc);
            } else {
                for (int k = 0; k < 4; ++k) {
                    const int col = c + 16 * k;
                    const __m512 w = decode16_ps(cb, codes + (col >> 1), sm + col);
                    acc[k] = _mm512_fmadd_ps(w, _mm512_loadu_ps(x + col), acc[k]);
                }
            }
        }
        float partial[kAccStripe];
        store_acc4(acc, partial);
        accum_encoded_range(cb, codes, sm, esc, x, partial, c, cols);
        y[r] = reduce_partials(partial);
    }
}

void gemv_bf16_avx512_range(const uint16_t* W, const float* x, float* y, int cols, int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
        const uint16_t* row = W + std::size_t(r) * std::size_t(cols);
        __m512 acc[4];
        for (int k = 0; k < 4; ++k)
            acc[k] = _mm512_setzero_ps();
        int c = 0;
        for (; c + kAccStripe <= cols; c += kAccStripe) {
            for (int k = 0; k < 4; ++k) {
                const int col = c + 16 * k;
                acc[k] =
                    _mm512_fmadd_ps(bf16_16_to_ps(row + col), _mm512_loadu_ps(x + col), acc[k]);
            }
        }
        float partial[kAccStripe];
        store_acc4(acc, partial);
        for (; c < cols; ++c) {
            partial[c & (kAccStripe - 1)] =
                exact_fma(bf16_to_fp32(row[c]), x[c], partial[c & (kAccStripe - 1)]);
        }
        y[r] = reduce_partials(partial);
    }
}

void gemv_f32_avx512_range(const float* W, const float* x, float* y, int cols, int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
        const float* row = W + std::size_t(r) * std::size_t(cols);
        __m512 acc[4];
        for (int k = 0; k < 4; ++k)
            acc[k] = _mm512_setzero_ps();
        int c = 0;
        for (; c + kAccStripe <= cols; c += kAccStripe) {
            for (int k = 0; k < 4; ++k) {
                const int col = c + 16 * k;
                acc[k] =
                    _mm512_fmadd_ps(_mm512_loadu_ps(row + col), _mm512_loadu_ps(x + col), acc[k]);
            }
        }
        float partial[kAccStripe];
        store_acc4(acc, partial);
        for (; c < cols; ++c) {
            partial[c & (kAccStripe - 1)] = exact_fma(row[c], x[c], partial[c & (kAccStripe - 1)]);
        }
        y[r] = reduce_partials(partial);
    }
}

double peak_fma_gflops_avx512() {
    __m512 acc[8];
    for (int k = 0; k < 8; ++k)
        acc[k] = _mm512_set1_ps(1.0f + 1e-4f * float(k));
    const __m512 scale = _mm512_set1_ps(1.0000001f);
    constexpr int kIters = 2000000;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) {
        acc[0] = _mm512_fmadd_ps(acc[0], scale, acc[0]);
        acc[1] = _mm512_fmadd_ps(acc[1], scale, acc[1]);
        acc[2] = _mm512_fmadd_ps(acc[2], scale, acc[2]);
        acc[3] = _mm512_fmadd_ps(acc[3], scale, acc[3]);
        acc[4] = _mm512_fmadd_ps(acc[4], scale, acc[4]);
        acc[5] = _mm512_fmadd_ps(acc[5], scale, acc[5]);
        acc[6] = _mm512_fmadd_ps(acc[6], scale, acc[6]);
        acc[7] = _mm512_fmadd_ps(acc[7], scale, acc[7]);
        asm volatile(""
                     :
                     : "v"(acc[0]), "v"(acc[1]), "v"(acc[2]), "v"(acc[3]), "v"(acc[4]), "v"(acc[5]),
                       "v"(acc[6]), "v"(acc[7])
                     : "memory");
    }
    const auto t1 = std::chrono::steady_clock::now();
    __m512 s = _mm512_setzero_ps();
    for (int k = 0; k < 8; ++k)
        s = _mm512_add_ps(s, acc[k]);
    alignas(64) float out[16];
    _mm512_store_ps(out, s);
    volatile float sink = out[0] + out[15];
    (void)sink;
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    const double flops = double(kIters) * 8.0 * 16.0 * 2.0;
    return flops / seconds / 1e9;
}

float stream_read_f32_avx512(const float* a, std::size_t n, int threads) {
    if (n % 64u)
        throw std::runtime_error("stream_read: n must be a multiple of 64");
    const std::size_t nblocks = n / 64u;
    float partials[256] = {};
    if (threads > 256)
        threads = 256;
#pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        const std::size_t b0 = (nblocks * std::size_t(tid)) / std::size_t(nt);
        const std::size_t b1 = (nblocks * std::size_t(tid + 1)) / std::size_t(nt);
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();
        for (std::size_t b = b0; b < b1; ++b) {
            const std::size_t i = b * 64u;
            acc0 = _mm512_add_ps(acc0, _mm512_loadu_ps(a + i));
            acc1 = _mm512_add_ps(acc1, _mm512_loadu_ps(a + i + 16));
            acc2 = _mm512_add_ps(acc2, _mm512_loadu_ps(a + i + 32));
            acc3 = _mm512_add_ps(acc3, _mm512_loadu_ps(a + i + 48));
        }
        const __m512 s = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        partials[tid] = _mm512_reduce_add_ps(s);
    }
    float total = 0.f;
    const int nt_cap = std::max(threads, 1);
    for (int i = 0; i < nt_cap; ++i)
        total += partials[i];
    return total;
}

void stream_triad_f32_avx512(float* a, const float* b, const float* c, std::size_t n, float scalar,
                             int threads, bool nontemporal) {
    if (n % 64u)
        throw std::runtime_error("stream_triad: n must be a multiple of 64");
    const __m512 s = _mm512_set1_ps(scalar);
    const std::size_t nvec = n / 16u;
#pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        const std::size_t v0 = (nvec * std::size_t(tid)) / std::size_t(nt);
        const std::size_t v1 = (nvec * std::size_t(tid + 1)) / std::size_t(nt);
        for (std::size_t v = v0; v < v1; ++v) {
            const std::size_t i = v * 16u;
            const __m512 y = _mm512_fmadd_ps(s, _mm512_loadu_ps(c + i), _mm512_loadu_ps(b + i));
            if (nontemporal) {
                _mm512_stream_ps(a + i, y);
            } else {
                _mm512_storeu_ps(a + i, y);
            }
        }
    }
    if (nontemporal)
        _mm_sfence();
}

void stream_copy_f32_avx512(float* dst, const float* src, std::size_t n, int threads) {
    if (n % 64u)
        throw std::runtime_error("stream_copy: n must be a multiple of 64");
    const std::size_t nvec = n / 16u;
#pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        const std::size_t v0 = (nvec * std::size_t(tid)) / std::size_t(nt);
        const std::size_t v1 = (nvec * std::size_t(tid + 1)) / std::size_t(nt);
        for (std::size_t v = v0; v < v1; ++v) {
            const std::size_t i = v * 16u;
            _mm512_storeu_ps(dst + i, _mm512_loadu_ps(src + i));
        }
    }
}

} // namespace expcode
