#include "kernels.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <thread>

#include <cpuid.h>
#include <omp.h>

namespace expcode {
namespace {

template <class Fn> void parallel_rows(int rows, int threads, Fn fn) {
    if (rows <= 0)
        return;
    if (threads < 1)
        threads = 1;
    if (threads > rows)
        threads = rows;
    if (threads == 1) {
        fn(0, rows);
        return;
    }
#pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        const int r0 = int((long long)rows * tid / nt);
        const int r1 = int((long long)rows * (tid + 1) / nt);
        fn(r0, r1);
    }
}

Backend resolve(Backend backend, const IsaInfo& isa) {
    if (backend == Backend::Auto) {
        if (isa.avx512f && isa.avx512bw)
            return Backend::Avx512;
        if (isa.avx2 && isa.fma)
            return Backend::Avx2;
        return Backend::Scalar;
    }
    if (backend == Backend::Avx512 && !(isa.avx512f && isa.avx512bw)) {
        throw std::runtime_error("expcode: AVX-512F/BW is not enabled on this CPU");
    }
    if (backend == Backend::Avx2 && !(isa.avx2 && isa.fma)) {
        throw std::runtime_error("expcode: AVX2/FMA is not enabled on this CPU");
    }
    return backend;
}

const IsaInfo& isa_cached() {
    static const IsaInfo info = detect_isa();
    return info;
}

} // namespace

IsaInfo detect_isa() {
    IsaInfo info;
    __builtin_cpu_init();
    info.avx2 = __builtin_cpu_supports("avx2");
    info.fma = __builtin_cpu_supports("fma");
    info.avx512f = __builtin_cpu_supports("avx512f");
    info.avx512bw = __builtin_cpu_supports("avx512bw");
    info.logical_cpus = std::max(1, int(std::thread::hardware_concurrency()));

    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        info.stepping = int(eax & 0xF);
        info.model = int((eax >> 4) & 0xF);
        info.family = int((eax >> 8) & 0xF);
        if (info.family == 6 || info.family == 15) {
            info.model += int((eax >> 16) & 0xF) << 4;
        }
        if (info.family == 15)
            info.family += int((eax >> 20) & 0xFF);
    }
    unsigned max_ext = __get_cpuid_max(0x80000000u, nullptr);
    if (max_ext >= 0x80000004u) {
        char* dst = info.brand;
        for (unsigned leaf = 0x80000002u; leaf <= 0x80000004u; ++leaf) {
            unsigned r[4] = {};
            __cpuid(leaf, r[0], r[1], r[2], r[3]);
            std::memcpy(dst, r, 16);
            dst += 16;
        }
        info.brand[48] = 0;
    }
    return info;
}

void gemv_bf16(const uint16_t* W, const float* x, float* y, int rows, int cols, Backend backend,
               int threads) {
    const Backend use = resolve(backend, isa_cached());
    parallel_rows(rows, threads, [&](int r0, int r1) {
        if (use == Backend::Avx512) {
            gemv_bf16_avx512_range(W, x, y, cols, r0, r1);
        } else if (use == Backend::Avx2) {
            gemv_bf16_avx2_range(W, x, y, cols, r0, r1);
        } else {
            gemv_bf16_scalar_range(W, x, y, cols, r0, r1);
        }
    });
}

void gemv_f32(const float* W, const float* x, float* y, int rows, int cols, Backend backend,
              int threads) {
    const Backend use = resolve(backend, isa_cached());
    parallel_rows(rows, threads, [&](int r0, int r1) {
        if (use == Backend::Avx512) {
            gemv_f32_avx512_range(W, x, y, cols, r0, r1);
        } else if (use == Backend::Avx2) {
            gemv_f32_avx2_range(W, x, y, cols, r0, r1);
        } else {
            gemv_f32_scalar_range(W, x, y, cols, r0, r1);
        }
    });
}

void gemv_encoded(const MatrixView& m, const float* x, float* y, Backend backend, int threads) {
    const Backend use = resolve(backend, isa_cached());
    parallel_rows(int(m.rows), threads, [&](int r0, int r1) {
        if (use == Backend::Avx512) {
            gemv_encoded_avx512_range(m, x, y, r0, r1);
        } else if (use == Backend::Avx2) {
            gemv_encoded_avx2_range(m, x, y, r0, r1);
        } else {
            gemv_encoded_scalar_range(m, x, y, r0, r1);
        }
    });
}

double peak_fma_gflops(Backend backend) {
    const Backend use = resolve(backend, isa_cached());
    if (use == Backend::Avx512)
        return peak_fma_gflops_avx512();
    if (use == Backend::Avx2)
        return peak_fma_gflops_avx2();
    throw std::runtime_error("expcode: peak FMA is only measured for AVX2 and AVX-512");
}

float stream_read_f32(const float* a, std::size_t n, int threads) {
    const IsaInfo& isa = isa_cached();
    if (isa.avx512f)
        return stream_read_f32_avx512(a, n, threads);
    if (n % 64u)
        throw std::runtime_error("stream_read: n must be a multiple of 64");
    float acc = 0.f;
#pragma omp parallel for reduction(+ : acc) schedule(static) num_threads(threads)
    for (std::size_t i = 0; i < n; ++i)
        acc += a[i];
    return acc;
}

void stream_triad_f32(float* a, const float* b, const float* c, std::size_t n, float scalar,
                      int threads, bool nontemporal) {
    const IsaInfo& isa = isa_cached();
    if (isa.avx512f) {
        stream_triad_f32_avx512(a, b, c, n, scalar, threads, nontemporal);
        return;
    }
    (void)nontemporal;
    if (n % 64u)
        throw std::runtime_error("stream_triad: n must be a multiple of 64");
#pragma omp parallel for schedule(static) num_threads(threads)
    for (std::size_t i = 0; i < n; ++i)
        a[i] = b[i] + scalar * c[i];
}

void stream_copy_f32(float* dst, const float* src, std::size_t n, int threads) {
    const IsaInfo& isa = isa_cached();
    if (isa.avx512f) {
        stream_copy_f32_avx512(dst, src, n, threads);
        return;
    }
    if (n % 64u)
        throw std::runtime_error("stream_copy: n must be a multiple of 64");
#pragma omp parallel for schedule(static) num_threads(threads)
    for (std::size_t i = 0; i < n; ++i)
        dst[i] = src[i];
}

} // namespace expcode
