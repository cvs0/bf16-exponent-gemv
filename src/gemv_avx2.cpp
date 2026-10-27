#include "kernels.hpp"

#include <chrono>
#include <cstring>

#include <immintrin.h>

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

inline __m256 decode8_ps(const uint8_t* codebook, const uint8_t* code_bytes, const uint8_t* sm) {
    const __m128i table = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codebook));
    unsigned packed = 0;
    std::memcpy(&packed, code_bytes, 4);
    const __m128i raw = _mm_cvtsi32_si128(static_cast<int>(packed));
    const __m128i nibble = _mm_set1_epi8(0x0F);
    const __m128i lo = _mm_and_si128(raw, nibble);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), nibble);
    const __m128i exps =
        _mm_unpacklo_epi8(_mm_shuffle_epi8(table, lo), _mm_shuffle_epi8(table, hi));
    const __m128i smv = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(sm));
    const __m256i sm32 = _mm256_cvtepu8_epi32(smv);
    const __m256i e32 = _mm256_cvtepu8_epi32(exps);
    const __m256i sign = _mm256_slli_epi32(_mm256_and_si256(sm32, _mm256_set1_epi32(0x80)), 24);
    const __m256i mant = _mm256_slli_epi32(_mm256_and_si256(sm32, _mm256_set1_epi32(0x7F)), 16);
    const __m256i eb = _mm256_slli_epi32(e32, 23);
    return _mm256_castsi256_ps(_mm256_or_si256(sign, _mm256_or_si256(mant, eb)));
}

inline __m256 bf16_8_to_ps(const uint16_t* p) {
    const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(v), 16));
}

inline void store_acc8(__m256 acc[8], float partial[kAccStripe]) {
    for (int k = 0; k < 8; ++k)
        _mm256_storeu_ps(partial + 8 * k, acc[k]);
}

inline void load_acc8(const float partial[kAccStripe], __m256 acc[8]) {
    for (int k = 0; k < 8; ++k)
        acc[k] = _mm256_loadu_ps(partial + 8 * k);
}

} // namespace

void gemv_encoded_avx2_range(const MatrixView& m, const float* x, float* y, int r0, int r1) {
    const int cols = int(m.cols);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* cb = m.codebook_for_row(uint32_t(r));
        const uint8_t* codes = m.codes + std::size_t(r) * m.code_stride;
        const uint8_t* sm = m.sign_mant + std::size_t(r) * m.sm_stride;
        const uint8_t* esc = m.escapes + m.escape_offset_at(uint32_t(r));
        __m256 acc[8];
        for (int k = 0; k < 8; ++k)
            acc[k] = _mm256_setzero_ps();
        int c = 0;
        for (; c + kAccStripe <= cols; c += kAccStripe) {
            if (block_has_escape(codes + (c >> 1))) {
                float partial[kAccStripe];
                store_acc8(acc, partial);
                accum_encoded_range(cb, codes, sm, esc, x, partial, c, c + kAccStripe);
                load_acc8(partial, acc);
            } else {
                for (int k = 0; k < 8; ++k) {
                    const int col = c + 8 * k;
                    const __m256 w = decode8_ps(cb, codes + (col >> 1), sm + col);
                    acc[k] = _mm256_fmadd_ps(w, _mm256_loadu_ps(x + col), acc[k]);
                }
            }
        }
        float partial[kAccStripe];
        store_acc8(acc, partial);
        accum_encoded_range(cb, codes, sm, esc, x, partial, c, cols);
        y[r] = reduce_partials(partial);
    }
}

void gemv_bf16_avx2_range(const uint16_t* W, const float* x, float* y, int cols, int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
        const uint16_t* row = W + std::size_t(r) * std::size_t(cols);
        __m256 acc[8];
        for (int k = 0; k < 8; ++k)
            acc[k] = _mm256_setzero_ps();
        int c = 0;
        for (; c + kAccStripe <= cols; c += kAccStripe) {
            for (int k = 0; k < 8; ++k) {
                const int col = c + 8 * k;
                acc[k] = _mm256_fmadd_ps(bf16_8_to_ps(row + col), _mm256_loadu_ps(x + col), acc[k]);
            }
        }
        float partial[kAccStripe];
        store_acc8(acc, partial);
        for (; c < cols; ++c) {
            partial[c & (kAccStripe - 1)] =
                exact_fma(bf16_to_fp32(row[c]), x[c], partial[c & (kAccStripe - 1)]);
        }
        y[r] = reduce_partials(partial);
    }
}

void gemv_f32_avx2_range(const float* W, const float* x, float* y, int cols, int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
        const float* row = W + std::size_t(r) * std::size_t(cols);
        __m256 acc[8];
        for (int k = 0; k < 8; ++k)
            acc[k] = _mm256_setzero_ps();
        int c = 0;
        for (; c + kAccStripe <= cols; c += kAccStripe) {
            for (int k = 0; k < 8; ++k) {
                const int col = c + 8 * k;
                acc[k] =
                    _mm256_fmadd_ps(_mm256_loadu_ps(row + col), _mm256_loadu_ps(x + col), acc[k]);
            }
        }
        float partial[kAccStripe];
        store_acc8(acc, partial);
        for (; c < cols; ++c) {
            partial[c & (kAccStripe - 1)] = exact_fma(row[c], x[c], partial[c & (kAccStripe - 1)]);
        }
        y[r] = reduce_partials(partial);
    }
}

double peak_fma_gflops_avx2() {
    __m256 acc[8];
    for (int k = 0; k < 8; ++k)
        acc[k] = _mm256_set1_ps(1.0f + 1e-4f * float(k));
    const __m256 scale = _mm256_set1_ps(1.0000001f);
    constexpr int kIters = 4000000;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) {
        acc[0] = _mm256_fmadd_ps(acc[0], scale, acc[0]);
        acc[1] = _mm256_fmadd_ps(acc[1], scale, acc[1]);
        acc[2] = _mm256_fmadd_ps(acc[2], scale, acc[2]);
        acc[3] = _mm256_fmadd_ps(acc[3], scale, acc[3]);
        acc[4] = _mm256_fmadd_ps(acc[4], scale, acc[4]);
        acc[5] = _mm256_fmadd_ps(acc[5], scale, acc[5]);
        acc[6] = _mm256_fmadd_ps(acc[6], scale, acc[6]);
        acc[7] = _mm256_fmadd_ps(acc[7], scale, acc[7]);
        asm volatile(""
                     :
                     : "v"(acc[0]), "v"(acc[1]), "v"(acc[2]), "v"(acc[3]), "v"(acc[4]), "v"(acc[5]),
                       "v"(acc[6]), "v"(acc[7])
                     : "memory");
    }
    const auto t1 = std::chrono::steady_clock::now();
    __m256 s = _mm256_setzero_ps();
    for (int k = 0; k < 8; ++k)
        s = _mm256_add_ps(s, acc[k]);
    alignas(32) float out[8];
    _mm256_store_ps(out, s);
    volatile float sink = out[0] + out[7];
    (void)sink;
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    const double flops = double(kIters) * 8.0 * 8.0 * 2.0;
    return flops / seconds / 1e9;
}

} // namespace expcode
