#pragma once

#include "expcode/gemv_ref.hpp"

namespace expcode {

enum class Backend {
    Scalar,
    Avx2,
    Avx512,
    Auto,
};

struct IsaInfo {
    bool avx2 = false;
    bool fma = false;
    bool avx512f = false;
    bool avx512bw = false;
    int logical_cpus = 1;
    int family = 0;
    int model = 0;
    int stepping = 0;
    char brand[49] = {};
};

IsaInfo detect_isa();

// x is binary32, already promoted from BF16. y[r] receives the canonical
// stripe dot of row r. threads <= 1 runs the range on the caller thread.
// A requested backend that the CPU cannot execute throws.
void gemv_bf16(const uint16_t* W, const float* x, float* y, int rows, int cols, Backend backend,
               int threads);
void gemv_f32(const float* W, const float* x, float* y, int rows, int cols, Backend backend,
              int threads);
void gemv_encoded(const MatrixView& m, const float* x, float* y, Backend backend, int threads);

// In-register FMA throughput, no memory traffic beyond the instruction
// stream. Returns 1e9 binary32 FLOP/s (each FMA counts as 2).
double peak_fma_gflops(Backend backend);

// Timed by the caller. Each returns a checksum so the loads cannot be
// deleted. n is a multiple of 64. threads is honored inside the call.
float stream_read_f32(const float* a, std::size_t n, int threads);
void stream_triad_f32(float* a, const float* b, const float* c, std::size_t n, float scalar,
                      int threads, bool nontemporal);
void stream_copy_f32(float* dst, const float* src, std::size_t n, int threads);

} // namespace expcode
