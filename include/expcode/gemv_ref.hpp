#pragma once

#include "expcode/codec.hpp"

namespace expcode {

// partial[c & 63] += w[c] * x[c] via IEEE-754 FMA, then a sequential sum of
// the 64 partials. See kAccStripe in format.hpp.

inline float reduce_partials(const float partial[kAccStripe]) {
    float y = 0.f;
    for (int i = 0; i < kAccStripe; ++i)
        y = exact_add(y, partial[i]);
    return y;
}

// Decode and accumulate columns [c0, c1) of one row.
// `esc` advances once per escape nibble. `partial` is the running stripe
// (index = column mod 64), not a fresh buffer for the range.
inline void accum_encoded_range(const uint8_t* codebook, const uint8_t* codes, const uint8_t* sm,
                                const uint8_t*& esc, const float* x, float partial[kAccStripe],
                                int c0, int c1) {
    for (int c = c0; c < c1; ++c) {
        const uint8_t code = code_at(codes, uint32_t(c));
        const uint8_t exponent = (code == uint8_t(kEscapeCode)) ? (*esc++) : codebook[code];
        const float w = bf16_to_fp32(bf16_pack(sm[c], exponent));
        partial[c & (kAccStripe - 1)] = exact_fma(w, x[c], partial[c & (kAccStripe - 1)]);
    }
}

inline void gemv_encoded_scalar_range(const MatrixView& m, const float* x, float* y, int r0,
                                      int r1) {
    for (int r = r0; r < r1; ++r) {
        float partial[kAccStripe] = {};
        const uint8_t* cb = m.codebook_for_row(uint32_t(r));
        const uint8_t* codes = m.codes + std::size_t(r) * m.code_stride;
        const uint8_t* sm = m.sign_mant + std::size_t(r) * m.sm_stride;
        const uint8_t* esc = m.escapes + m.escape_offset_at(uint32_t(r));
        accum_encoded_range(cb, codes, sm, esc, x, partial, 0, int(m.cols));
        y[r] = reduce_partials(partial);
    }
}

inline void gemv_bf16_scalar_range(const uint16_t* W, const float* x, float* y, int cols, int r0,
                                   int r1) {
    for (int r = r0; r < r1; ++r) {
        float partial[kAccStripe] = {};
        const uint16_t* row = W + std::size_t(r) * std::size_t(cols);
        for (int c = 0; c < cols; ++c) {
            partial[c & (kAccStripe - 1)] =
                exact_fma(bf16_to_fp32(row[c]), x[c], partial[c & (kAccStripe - 1)]);
        }
        y[r] = reduce_partials(partial);
    }
}

inline void gemv_f32_scalar_range(const float* W, const float* x, float* y, int cols, int r0,
                                  int r1) {
    for (int r = r0; r < r1; ++r) {
        float partial[kAccStripe] = {};
        const float* row = W + std::size_t(r) * std::size_t(cols);
        for (int c = 0; c < cols; ++c) {
            partial[c & (kAccStripe - 1)] = exact_fma(row[c], x[c], partial[c & (kAccStripe - 1)]);
        }
        y[r] = reduce_partials(partial);
    }
}

// Left-to-right FMA chain. This is NOT the GEMV contract. It exists so tests
// can show that a different summation order changes binary32 bits.
inline float dot_bf16_sequential(const uint16_t* w, const float* x, int cols) {
    float acc = 0.f;
    for (int c = 0; c < cols; ++c)
        acc = exact_fma(bf16_to_fp32(w[c]), x[c], acc);
    return acc;
}

inline void promote_bf16(const uint16_t* src, float* dst, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        dst[i] = bf16_to_fp32(src[i]);
}

} // namespace expcode
