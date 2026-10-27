#include "expcode/expcode.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

int g_fails = 0;

void check(bool cond, const char* expr, const char* file, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, expr);
        ++g_fails;
    }
}

#define CHECK(cond) check(static_cast<bool>(cond), #cond, __FILE__, __LINE__)

uint64_t xorshift(uint64_t& s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

std::vector<float> promote_x(const std::vector<uint16_t>& x) {
    std::vector<float> out(x.size());
    expcode::promote_bf16(x.data(), out.data(), x.size());
    return out;
}

bool same_vec(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!expcode::float_bits_equal(a[i], b[i]))
            return false;
    }
    return true;
}

void expect_gemv_match(const std::vector<uint16_t>& w, int rows, int cols,
                       const std::vector<uint16_t>& x, uint32_t group_rows) {
    const auto enc = expcode::EncodedMatrix::encode(w, uint32_t(rows), uint32_t(cols), group_rows);
    std::vector<uint16_t> roundtrip(w.size());
    enc.decode(roundtrip);
    CHECK(roundtrip == w);
    CHECK(enc.bytes().size() == expcode::encoded_storage_bytes(uint32_t(rows), uint32_t(cols),
                                                               enc.n_groups(), enc.n_escape()));

    const std::vector<float> xf = promote_x(x);
    std::vector<float> y_ref(rows), y_enc(rows), y_f32(rows);
    expcode::gemv_bf16_scalar_range(w.data(), xf.data(), y_ref.data(), cols, 0, rows);
    expcode::gemv_encoded_scalar_range(enc.view(), xf.data(), y_enc.data(), 0, rows);
    CHECK(same_vec(y_ref, y_enc));

    std::vector<float> wf(w.size());
    expcode::promote_bf16(w.data(), wf.data(), w.size());
    expcode::gemv_f32_scalar_range(wf.data(), xf.data(), y_f32.data(), cols, 0, rows);
    CHECK(same_vec(y_ref, y_f32));

    const expcode::IsaInfo isa = expcode::detect_isa();
    if (isa.avx2 && isa.fma) {
        std::vector<float> y(rows);
        expcode::gemv_bf16(w.data(), xf.data(), y.data(), rows, cols, expcode::Backend::Avx2, 1);
        CHECK(same_vec(y_ref, y));
        expcode::gemv_encoded(enc.view(), xf.data(), y.data(), expcode::Backend::Avx2, 1);
        CHECK(same_vec(y_ref, y));
        expcode::gemv_f32(wf.data(), xf.data(), y.data(), rows, cols, expcode::Backend::Avx2, 1);
        CHECK(same_vec(y_ref, y));
        if (rows >= 2) {
            expcode::gemv_encoded(enc.view(), xf.data(), y.data(), expcode::Backend::Avx2, 4);
            CHECK(same_vec(y_ref, y));
        }
    }
    if (isa.avx512f && isa.avx512bw) {
        std::vector<float> y(rows);
        expcode::gemv_bf16(w.data(), xf.data(), y.data(), rows, cols, expcode::Backend::Avx512, 1);
        CHECK(same_vec(y_ref, y));
        expcode::gemv_encoded(enc.view(), xf.data(), y.data(), expcode::Backend::Avx512, 1);
        CHECK(same_vec(y_ref, y));
        expcode::gemv_f32(wf.data(), xf.data(), y.data(), rows, cols, expcode::Backend::Avx512, 1);
        CHECK(same_vec(y_ref, y));
        expcode::gemv_encoded(enc.view(), xf.data(), y.data(), expcode::Backend::Avx512, 3);
        CHECK(same_vec(y_ref, y));
    }
}

void test_bf16_constants() {
    CHECK(expcode::fp32_to_bf16(1.f) == 0x3F80);
    CHECK(expcode::fp32_to_bf16(-1.f) == 0xBF80);
    CHECK(expcode::fp32_to_bf16(2.f) == 0x4000);
    CHECK(expcode::fp32_to_bf16(0.f) == 0x0000);
    CHECK(expcode::float_bits_equal(expcode::bf16_to_fp32(0x3F80), 1.f));
    CHECK(expcode::bf16_pack(expcode::bf16_sign_mant(0xBF80), expcode::bf16_exponent(0xBF80)) ==
          0xBF80);
}

void test_codebook_tie_break() {
    uint64_t hist[256] = {};
    hist[9] = 4;
    hist[5] = 4;
    hist[20] = 1;
    const expcode::Codebook cb = expcode::codebook_from_histogram(hist);
    CHECK(cb.exp[0] == 5);
    CHECK(cb.exp[1] == 9);
    CHECK(cb.exp[2] == 20);
    CHECK(cb.inv[5] == 0);
    CHECK(cb.inv[9] == 1);
    CHECK(cb.inv[7] == 255);
}

void test_nibble_layout() {
    // 32 equal-frequency exponents. Ties keep the smaller exponent, so codes
    // 0..14 map to exponents 0..14 and the rest escape.
    std::vector<uint16_t> w(32);
    for (int i = 0; i < 32; ++i)
        w[i] = expcode::bf16_pack(uint8_t(i & 0x7F), uint8_t(i));
    const auto enc = expcode::EncodedMatrix::encode(w, 1, 32, 1);
    const expcode::MatrixView m = enc.view();
    CHECK(m.codebook[0] == 0);
    CHECK(m.codebook[14] == 14);
    CHECK(expcode::code_at(m.codes, 0) == 0);
    CHECK(expcode::code_at(m.codes, 1) == 1);
    CHECK(m.codes[0] == 0x10);
    CHECK(expcode::code_at(m.codes, 15) == expcode::kEscapeCode);
    CHECK(enc.n_escape() == 17);
    std::vector<uint16_t> out(32);
    enc.decode(out);
    CHECK(out == w);
}

void test_all_bf16_patterns() {
    std::vector<uint16_t> w(65536);
    for (int i = 0; i < 65536; ++i)
        w[i] = uint16_t(i);
    expect_gemv_match(w, 1, 65536, std::vector<uint16_t>(65536, 0x3F80), 0);
    // Column vector, one codebook per row, forces every exponent through both
    // the direct path and, for rare rows, the escape path.
    expect_gemv_match(w, 65536, 1, std::vector<uint16_t>(1, 0x3F80), 1);
}

void test_shapes_and_groups() {
    uint64_t rng = 0x123456789ABCDEF0ull;
    const int shapes[][2] = {{1, 1},  {1, 2},   {3, 3},   {4, 7},    {5, 8},   {2, 15},
                             {7, 16}, {3, 17},  {4, 31},  {4, 32},   {8, 63},  {6, 64},
                             {5, 65}, {9, 127}, {8, 128}, {16, 192}, {13, 768}};
    for (const auto& sh : shapes) {
        const int rows = sh[0];
        const int cols = sh[1];
        std::vector<uint16_t> w(std::size_t(rows) * cols);
        std::vector<uint16_t> x(cols);
        for (uint16_t& v : w)
            v = uint16_t(xorshift(rng));
        for (uint16_t& v : x)
            v = uint16_t(xorshift(rng) & 0xFFFF);
        expect_gemv_match(w, rows, cols, x, 0);
        expect_gemv_match(w, rows, cols, x, 1);
        if (rows >= 4)
            expect_gemv_match(w, rows, cols, x, 3);
    }
}

void test_fast_path_distribution() {
    // Ten exponents, no escapes. This is the decode kernel's fast path.
    uint64_t rng = 42;
    const int rows = 32;
    const int cols = 256;
    const uint8_t exps[10] = {120, 121, 122, 123, 124, 125, 126, 127, 128, 129};
    std::vector<uint16_t> w(std::size_t(rows) * cols);
    std::vector<uint16_t> x(cols);
    for (std::size_t i = 0; i < w.size(); ++i) {
        const uint8_t e = exps[xorshift(rng) % 10];
        const uint8_t sm = uint8_t(xorshift(rng) & 0xFF);
        w[i] = expcode::bf16_pack(sm, e);
    }
    for (uint16_t& v : x)
        v = expcode::bf16_pack(uint8_t(xorshift(rng) & 0x7F), 127);
    expect_gemv_match(w, rows, cols, x, 0);
    const auto enc = expcode::EncodedMatrix::encode(w, rows, cols, 0);
    CHECK(enc.n_escape() == 0);
}

void test_single_escape_in_second_block() {
    // Fifteen exponents fill the codebook. Exponent 10 occurs once, at column
    // 70, which is inside the second 64-column stripe. The fast path must
    // take the scalar repair for that stripe only.
    const int cols = 128;
    std::vector<uint16_t> w(cols);
    for (int c = 0; c < cols; ++c) {
        w[c] = expcode::bf16_pack(uint8_t(c & 0x7F), uint8_t(100 + (c % 15)));
    }
    w[70] = expcode::bf16_pack(0x3, 10);
    std::vector<uint16_t> x(cols, 0x3F80);
    const auto enc = expcode::EncodedMatrix::encode(w, 1, cols, 0);
    CHECK(enc.n_escape() == 1);
    CHECK(expcode::code_at(enc.view().codes, 70) == expcode::kEscapeCode);
    expect_gemv_match(w, 1, cols, x, 0);
}

void test_order_matters() {
    // Stripe partial 0 sees 1e20 and -1e20. Partial 1 sees 1.
    // Sequential accumulation loses the 1.
    const int cols = 128;
    std::vector<uint16_t> w(cols, 0);
    std::vector<float> x(cols, 0.f);
    w[0] = expcode::fp32_to_bf16(1e20f);
    w[1] = expcode::fp32_to_bf16(1.f);
    w[64] = expcode::fp32_to_bf16(-1e20f);
    x[0] = 1.f;
    x[1] = 1.f;
    x[64] = 1.f;
    // The products are formed from BF16-promoted values, so rebuild x from
    // the BF16 rounding of those magnitudes.
    w[0] = expcode::fp32_to_bf16(1e20f);
    w[1] = expcode::fp32_to_bf16(1.f);
    w[64] = expcode::fp32_to_bf16(-1e20f);
    float y_seq = expcode::dot_bf16_sequential(w.data(), x.data(), cols);
    float partial[expcode::kAccStripe] = {};
    for (int c = 0; c < cols; ++c) {
        partial[c & 63] = expcode::exact_fma(expcode::bf16_to_fp32(w[c]), x[c], partial[c & 63]);
    }
    const float y_stripe = expcode::reduce_partials(partial);
    CHECK(!expcode::float_bits_equal(y_seq, y_stripe));
    std::vector<float> y(1);
    expcode::gemv_bf16_scalar_range(w.data(), x.data(), y.data(), cols, 0, 1);
    CHECK(expcode::float_bits_equal(y[0], y_stripe));
}

void test_subnormals_and_infinities() {
    std::vector<uint16_t> w;
    std::vector<uint16_t> x;
    // Subnormals (exp 0), zeros, a few finite values, infinities.
    for (int mant = 0; mant < 128; ++mant) {
        w.push_back(expcode::bf16_pack(uint8_t(mant), 0));
        w.push_back(expcode::bf16_pack(uint8_t(0x80 | mant), 0));
    }
    w.push_back(0x7F80); // +inf
    w.push_back(0xFF80); // -inf
    w.push_back(0x3F80);
    x.assign(w.size(), 0x3F80);
    // inf * 1 is inf. Subnormal * 1 keeps the subnormal. No 0*inf.
    expect_gemv_match(w, 1, int(w.size()), x, 0);
}

void test_blob_roundtrip_and_reject() {
    std::vector<uint16_t> w = {0x3F80, 0x4000, 0xBF80, 0x0000, 0x3F00};
    const auto enc = expcode::EncodedMatrix::encode(w, 1, 5, 0);
    const auto copy = expcode::EncodedMatrix::parse_copy(enc.bytes());
    std::vector<uint16_t> out(5);
    copy.decode(out);
    CHECK(out == w);
    auto bad = std::vector<uint8_t>(enc.bytes().begin(), enc.bytes().end());
    bad[0] ^= 0xFF;
    bool threw = false;
    try {
        (void)expcode::EncodedMatrix::parse_copy(bad);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

void test_negative_zero_sign() {
    const uint16_t nz = 0x8000;
    CHECK(expcode::bf16_sign_mant(nz) == 0x80);
    CHECK(expcode::bf16_pack(0x80, 0) == nz);
    std::vector<uint16_t> w = {nz, 0x0000, nz};
    std::vector<uint16_t> x = {0x3F80, 0x3F80, 0x3F80};
    expect_gemv_match(w, 1, 3, x, 0);
}

} // namespace

int main() {
    expcode::set_ieee_fp_env();
    test_bf16_constants();
    test_codebook_tie_break();
    test_nibble_layout();
    test_all_bf16_patterns();
    test_shapes_and_groups();
    test_fast_path_distribution();
    test_single_escape_in_second_block();
    test_order_matters();
    test_subnormals_and_infinities();
    test_blob_roundtrip_and_reject();
    test_negative_zero_sign();
    if (g_fails) {
        std::fprintf(stderr, "%d checks failed\n", g_fails);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
