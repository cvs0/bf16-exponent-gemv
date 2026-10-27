#include "expcode/gemv.hpp"
#include "model.hpp"
#include "pack.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Json {
    std::string s;
    std::vector<bool> need_comma;
    Json() {
        s = "{\n";
        need_comma.push_back(false);
    }
    void comma() {
        if (need_comma.back())
            s += ",\n";
        need_comma.back() = true;
    }
    std::string indent() const { return std::string(need_comma.size() * 2, ' '); }
    void raw_field(const std::string& key, const std::string& raw) {
        comma();
        s += indent() + "\"" + key + "\": " + raw;
    }
    void str_field(const std::string& key, const std::string& value) {
        std::string esc;
        for (unsigned char c : value) {
            if (c == '\\' || c == '"')
                esc.push_back('\\');
            if (c == '\n')
                esc += "\\n";
            else
                esc.push_back(char(c));
        }
        raw_field(key, "\"" + esc + "\"");
    }
    void num_field(const std::string& key, double v) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.9g", v);
        raw_field(key, buf);
    }
    void int_field(const std::string& key, long long v) { raw_field(key, std::to_string(v)); }
    void bool_field(const std::string& key, bool v) { raw_field(key, v ? "true" : "false"); }
    void begin_object_key(const std::string& key) {
        comma();
        s += indent() + "\"" + key + "\": {\n";
        need_comma.push_back(false);
    }
    void begin_array(const std::string& key) {
        comma();
        s += indent() + "\"" + key + "\": [\n";
        need_comma.push_back(false);
    }
    void begin_object() {
        comma();
        s += indent() + "{\n";
        need_comma.push_back(false);
    }
    void end_object() {
        s += "\n" + std::string((need_comma.size() - 1) * 2, ' ') + "}";
        need_comma.pop_back();
        need_comma.back() = true;
    }
    void end_array() {
        s += "\n" + std::string((need_comma.size() - 1) * 2, ' ') + "]";
        need_comma.pop_back();
        need_comma.back() = true;
    }
    std::string finish() const { return s + "\n}\n"; }
};

struct Sample {
    double min_s = 0;
    double median_s = 0;
    int reps = 1;
};

template <class Fn> Sample time_call(Fn&& fn, int trials, double target_seconds) {
    fn();
    const auto p0 = std::chrono::steady_clock::now();
    fn();
    const double probe =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - p0).count();
    int reps = 1;
    if (probe > 0 && probe < target_seconds) {
        reps = std::min(100, std::max(1, int(target_seconds / probe)));
    }
    std::vector<double> samples;
    for (int t = 0; t < trials; ++t) {
        const auto a = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i)
            fn();
        samples.push_back(
            std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count() / reps);
    }
    std::sort(samples.begin(), samples.end());
    return Sample{samples.front(), samples[samples.size() / 2], reps};
}

struct Aligned {
    float* p = nullptr;
    std::size_t n = 0;
    explicit Aligned(std::size_t n_in, float fill = 0.f) : n(n_in) {
        const std::size_t bytes = std::max(n * sizeof(float), std::size_t(64));
        if (posix_memalign(reinterpret_cast<void**>(&p), 64, bytes) != 0)
            throw std::bad_alloc();
        std::fill(p, p + n, fill);
    }
    Aligned(const Aligned&) = delete;
    Aligned& operator=(const Aligned&) = delete;
    ~Aligned() { std::free(p); }
};

long long mem_kib(const char* key) {
    std::ifstream in("/proc/meminfo");
    std::string k, unit;
    long long v = 0;
    while (in >> k >> v >> unit) {
        if (k == key)
            return v;
    }
    return 0;
}

std::string cpu_mhz() {
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("cpu MHz", 0) == 0) {
            const auto pos = line.find(':');
            if (pos != std::string::npos) {
                std::string v = line.substr(pos + 1);
                while (!v.empty() && v[0] == ' ')
                    v.erase(v.begin());
                return v;
            }
        }
    }
    return "";
}

const char* backend_name(expcode::Backend b) {
    if (b == expcode::Backend::Avx512)
        return "avx512";
    if (b == expcode::Backend::Avx2)
        return "avx2";
    return "scalar";
}

bool bits_equal(const std::vector<float>& a, const std::vector<float>& b, int* mismatch) {
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!expcode::float_bits_equal(a[i], b[i])) {
            if (mismatch)
                *mismatch = int(i);
            return false;
        }
    }
    return true;
}

uint64_t fnv_floats(const float* v, std::size_t n) {
    uint64_t h = 14695981039346656037ull;
    for (std::size_t i = 0; i < n; ++i) {
        const uint32_t b = expcode::float_bits(v[i]);
        for (int k = 0; k < 4; ++k) {
            h ^= uint64_t((b >> (8 * k)) & 0xFFu);
            h *= 1099511628211ull;
        }
    }
    return h;
}

uint64_t xorshift(uint64_t& s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

struct GemvCase {
    std::string name;
    int rows = 0;
    int cols = 0;
    std::vector<uint16_t> bf16;
};

GemvCase take_case(std::map<std::string, PackedTensor>& idx, const std::string& name) {
    auto it = idx.find(name);
    if (it == idx.end())
        throw std::runtime_error("missing " + name);
    GemvCase g;
    g.name = name;
    if (it->second.dims.size() != 2)
        throw std::runtime_error("not a matrix " + name);
    g.rows = int(it->second.dims[0]);
    g.cols = int(it->second.dims[1]);
    g.bf16 = std::move(it->second.bf16);
    idx.erase(it);
    return g;
}

void emit_gemv(Json& j, const std::string& model, const std::string& name, int rows, int cols,
               const char* storage, const char* isa, int threads, uint32_t group_rows,
               const Sample& sample, double weight_bytes) {
    const double flops = 2.0 * double(rows) * double(cols);
    const double bf16_bytes = 2.0 * double(rows) * double(cols);
    j.begin_object();
    j.str_field("model", model);
    j.str_field("name", name);
    j.int_field("rows", rows);
    j.int_field("cols", cols);
    j.str_field("storage", storage);
    j.str_field("isa", isa);
    j.int_field("threads", threads);
    j.int_field("group_rows", group_rows);
    j.num_field("seconds_min", sample.min_s);
    j.num_field("seconds_median", sample.median_s);
    j.int_field("reps", sample.reps);
    j.num_field("weight_bytes", weight_bytes);
    j.num_field("gflops", flops / sample.min_s / 1e9);
    j.num_field("actual_GBps", weight_bytes / sample.min_s / 1e9);
    j.num_field("effective_bf16_GBps", bf16_bytes / sample.min_s / 1e9);
    j.bool_field("bit_exact_vs_bf16", true);
    j.end_object();
    std::fprintf(stderr, "  %-8s %-7s th %d  %7.2f GFLOP/s  actual %6.2f GB/s  eff %6.2f GB/s\n",
                 storage, isa, threads, flops / sample.min_s / 1e9,
                 weight_bytes / sample.min_s / 1e9, bf16_bytes / sample.min_s / 1e9);
}

void bench_matrix(Json& j, const std::string& model, const GemvCase& g, const expcode::IsaInfo& isa,
                  const std::vector<int>& threads, bool with_f32, bool with_scalar,
                  bool group_sweep) {
    std::fprintf(stderr, "gemv %s %s %dx%d (%.2f MiB bf16)\n", model.c_str(), g.name.c_str(),
                 g.rows, g.cols, double(g.bf16.size()) * 2 / (1024.0 * 1024.0));
    std::fflush(stderr);
    const auto enc = expcode::EncodedMatrix::encode(g.bf16, uint32_t(g.rows), uint32_t(g.cols), 0);
    std::fprintf(stderr, "  encoded %.3f bpw  escapes %llu\n", enc.bits_per_weight(),
                 (unsigned long long)enc.n_escape());
    std::vector<uint16_t> xbf(g.cols);
    uint64_t seed = 0xC0FFEEULL ^ (uint64_t(g.rows) << 32) ^ uint64_t(g.cols);
    for (uint16_t& v : xbf) {
        v = expcode::bf16_pack(uint8_t(xorshift(seed) & 0xFF),
                               uint8_t(120 + (xorshift(seed) % 10)));
    }
    std::vector<float> x(g.cols);
    expcode::promote_bf16(xbf.data(), x.data(), x.size());
    std::vector<float> y_bf(g.rows), y_ex(g.rows);
    const expcode::Backend ref =
        (isa.avx512f && isa.avx512bw) ? expcode::Backend::Avx512 : expcode::Backend::Avx2;
    expcode::gemv_bf16(g.bf16.data(), x.data(), y_bf.data(), g.rows, g.cols, ref, 1);
    expcode::gemv_encoded(enc.view(), x.data(), y_ex.data(), ref, 1);
    int mismatch = -1;
    const std::vector<float> y_ref = y_bf;
    if (!bits_equal(y_bf, y_ex, &mismatch)) {
        std::fprintf(stderr, "MISMATCH row %d bf16 %08x exp %08x\n", mismatch,
                     expcode::float_bits(y_bf[mismatch]), expcode::float_bits(y_ex[mismatch]));
        throw std::runtime_error("GEMV bit mismatch");
    }
    std::vector<float> wf;
    if (with_f32) {
        wf.resize(g.bf16.size());
        expcode::promote_bf16(g.bf16.data(), wf.data(), wf.size());
        std::vector<float> y_f(g.rows);
        expcode::gemv_f32(wf.data(), x.data(), y_f.data(), g.rows, g.cols, ref, 1);
        if (!bits_equal(y_bf, y_f, &mismatch))
            throw std::runtime_error("FP32 GEMV bit mismatch");
    }
    std::vector<expcode::Backend> isas;
    if (with_scalar)
        isas.push_back(expcode::Backend::Scalar);
    if (isa.avx2 && isa.fma)
        isas.push_back(expcode::Backend::Avx2);
    if (isa.avx512f && isa.avx512bw)
        isas.push_back(expcode::Backend::Avx512);
    for (int th : threads) {
        for (expcode::Backend b : isas) {
            if (b == expcode::Backend::Scalar && th != 1)
                continue;
            const Sample sb = time_call(
                [&] {
                    expcode::gemv_bf16(g.bf16.data(), x.data(), y_bf.data(), g.rows, g.cols, b, th);
                },
                3, 0.12);
            const Sample se = time_call(
                [&] { expcode::gemv_encoded(enc.view(), x.data(), y_ex.data(), b, th); }, 3, 0.12);
            if (!bits_equal(y_ref, y_ex, &mismatch)) {
                throw std::runtime_error("threaded expcode GEMV bit mismatch");
            }
            emit_gemv(j, model, g.name, g.rows, g.cols, "bf16", backend_name(b), th,
                      uint32_t(g.rows), sb, 2.0 * g.rows * g.cols);
            emit_gemv(j, model, g.name, g.rows, g.cols, "expcode", backend_name(b), th,
                      uint32_t(g.rows), se, double(enc.bytes().size()));
            if (with_f32) {
                const Sample sf = time_call(
                    [&] {
                        expcode::gemv_f32(wf.data(), x.data(), y_bf.data(), g.rows, g.cols, b, th);
                    },
                    3, 0.12);
                emit_gemv(j, model, g.name, g.rows, g.cols, "fp32", backend_name(b), th,
                          uint32_t(g.rows), sf, 4.0 * g.rows * g.cols);
            }
        }
    }
    if (group_sweep && isa.avx512f) {
        for (uint32_t gr : {uint32_t(1), uint32_t(16)}) {
            const auto eg =
                expcode::EncodedMatrix::encode(g.bf16, uint32_t(g.rows), uint32_t(g.cols), gr);
            expcode::gemv_encoded(eg.view(), x.data(), y_ex.data(), expcode::Backend::Avx512, 1);
            if (!bits_equal(y_ref, y_ex, &mismatch))
                throw std::runtime_error("group sweep mismatch");
            for (int th : {1, threads.back()}) {
                const Sample se = time_call(
                    [&] {
                        expcode::gemv_encoded(eg.view(), x.data(), y_ex.data(),
                                              expcode::Backend::Avx512, th);
                    },
                    3, 0.12);
                emit_gemv(j, model, g.name, g.rows, g.cols, "expcode", "avx512", th,
                          eg.group_rows(), se, double(eg.bytes().size()));
            }
        }
    }
}

void bench_stream(Json& j, int threads_max) {
    j.begin_array("stream");
    const long long avail = mem_kib("MemAvailable:");
    std::vector<std::size_t> sizes = {16ull << 20, 64ull << 20, 256ull << 20, 1024ull << 20};
    if (avail > 7ll * 1024 * 1024)
        sizes.push_back(2048ull << 20);
    for (std::size_t bytes : sizes) {
        const std::size_t n = (bytes / sizeof(float)) & ~std::size_t(63);
        auto fits = [&](int arrays) {
            const long long need = (long long)n * 4 * arrays + (128ll << 20);
            return need < avail * 1024 * 3 / 4;
        };
        try {
            if (fits(1)) {
                Aligned a(n, 1.f);
                for (int th = 1; th <= threads_max; ++th) {
                    const Sample s = time_call(
                        [&] {
                            volatile float sink = expcode::stream_read_f32(a.p, n, th);
                            (void)sink;
                        },
                        5, 0.2);
                    j.begin_object();
                    j.str_field("kernel", "read");
                    j.int_field("bytes", (long long)n * 4);
                    j.int_field("threads", th);
                    j.num_field("seconds_min", s.min_s);
                    j.num_field("GBps", double(n) * 4.0 / s.min_s / 1e9);
                    j.end_object();
                    std::fprintf(stderr, "read %.0f MiB th %d  %.2f GB/s\n",
                                 double(n) * 4 / (1024 * 1024), th,
                                 double(n) * 4.0 / s.min_s / 1e9);
                }
            }
            if (fits(2)) {
                Aligned dst(n, 0.f);
                Aligned src(n, 1.f);
                for (int th = 1; th <= threads_max; ++th) {
                    const Sample s =
                        time_call([&] { expcode::stream_copy_f32(dst.p, src.p, n, th); }, 5, 0.2);
                    j.begin_object();
                    j.str_field("kernel", "copy");
                    j.int_field("bytes", (long long)n * 8);
                    j.int_field("threads", th);
                    j.num_field("seconds_min", s.min_s);
                    j.num_field("GBps", double(n) * 8.0 / s.min_s / 1e9);
                    j.end_object();
                }
            }
            if (fits(3)) {
                Aligned a(n, 0.f);
                Aligned b(n, 1.f);
                Aligned c(n, 2.f);
                for (int th = 1; th <= threads_max; ++th) {
                    const Sample s = time_call(
                        [&] { expcode::stream_triad_f32(a.p, b.p, c.p, n, 1.25f, th, false); }, 5,
                        0.2);
                    const Sample sn = time_call(
                        [&] { expcode::stream_triad_f32(a.p, b.p, c.p, n, 1.25f, th, true); }, 5,
                        0.2);
                    j.begin_object();
                    j.str_field("kernel", "triad");
                    j.str_field("store", "normal");
                    j.int_field("bytes", (long long)n * 12);
                    j.int_field("threads", th);
                    j.num_field("seconds_min", s.min_s);
                    j.num_field("GBps", double(n) * 12.0 / s.min_s / 1e9);
                    j.end_object();
                    j.begin_object();
                    j.str_field("kernel", "triad");
                    j.str_field("store", "nontemporal");
                    j.int_field("bytes", (long long)n * 12);
                    j.int_field("threads", th);
                    j.num_field("seconds_min", sn.min_s);
                    j.num_field("GBps", double(n) * 12.0 / sn.min_s / 1e9);
                    j.end_object();
                    std::fprintf(stderr, "triad %.0f MiB th %d  normal %.2f  nt %.2f GB/s\n",
                                 double(n) * 12 / (1024 * 1024), th,
                                 double(n) * 12.0 / s.min_s / 1e9,
                                 double(n) * 12.0 / sn.min_s / 1e9);
                }
            }
        } catch (const std::bad_alloc&) {
            std::fprintf(stderr, "stream alloc failed at %zu bytes\n", bytes);
        }
        std::fflush(stderr);
    }
    j.end_array();
}

int token_at(int pos, int vocab) {
    return (pos * 17 + 3) % vocab;
}

void bench_decode(Json& j, const Model& model, const std::string& model_name, expcode::Backend isa,
                  const std::vector<int>& threads, int prefill, int steps) {
    const int max_pos = prefill + steps;
    DecodeWorkspace ref_ws = make_workspace(model, max_pos);
    std::vector<float> ref_logits;
    for (int pos = 0; pos < max_pos; ++pos) {
        StepStats st = decode_step(model, ref_ws, token_at(pos, model.n_vocab), pos, isa, 1, true);
        if (pos + 1 == max_pos)
            ref_logits = std::move(st.logits);
    }
    for (bool encoded : {false, true}) {
        for (int th : threads) {
            DecodeWorkspace ws = make_workspace(model, max_pos);
            for (int pos = 0; pos < prefill; ++pos) {
                (void)decode_step(model, ws, token_at(pos, model.n_vocab), pos, isa, th, encoded);
            }
            struct Rec {
                double step;
                double gemv;
            };
            std::vector<Rec> recs;
            std::vector<float> last;
            for (int s = 0; s < steps; ++s) {
                const int pos = prefill + s;
                const auto t0 = std::chrono::steady_clock::now();
                StepStats st =
                    decode_step(model, ws, token_at(pos, model.n_vocab), pos, isa, th, encoded);
                recs.push_back(Rec{
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                    st.gemv_seconds});
                if (s + 1 == steps)
                    last = std::move(st.logits);
            }
            std::sort(recs.begin(), recs.end(),
                      [](const Rec& a, const Rec& b) { return a.step < b.step; });
            const Rec med = recs[recs.size() / 2];
            int mismatch = -1;
            if (!bits_equal(ref_logits, last, &mismatch)) {
                std::fprintf(stderr, "logits mismatch %s encoded=%d th=%d idx=%d\n",
                             model_name.c_str(), int(encoded), th, mismatch);
                throw std::runtime_error("decode logits are not bit-identical");
            }
            j.begin_object();
            j.str_field("model", model_name);
            j.str_field("storage", encoded ? "expcode" : "bf16");
            j.str_field("isa", backend_name(isa));
            j.int_field("threads", th);
            j.int_field("prefill", prefill);
            j.int_field("steps", steps);
            j.num_field("seconds_median", med.step);
            j.num_field("tok_per_s", 1.0 / med.step);
            j.num_field("gemv_seconds_of_median_step", med.gemv);
            j.num_field("gemv_fraction", med.gemv / med.step);
            char hash[32];
            std::snprintf(hash, sizeof(hash), "%016llx",
                          (unsigned long long)fnv_floats(last.data(), last.size()));
            j.str_field("logits_fnv", hash);
            j.bool_field("logits_bit_exact", true);
            j.end_object();
            std::fprintf(stderr, "decode %s %s th %d  %.3f tok/s  gemv %.1f%%\n",
                         model_name.c_str(), encoded ? "expcode" : "bf16", th, 1.0 / med.step,
                         100.0 * med.gemv / med.step);
            std::fflush(stderr);
        }
    }
}

void add_caches(Json& j) {
    j.begin_array("caches");
    for (int i = 0; i < 8; ++i) {
        const std::string base =
            "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(i) + "/";
        std::ifstream level(base + "level");
        if (!level)
            break;
        std::string lv, tp, sz;
        std::getline(level, lv);
        std::getline(std::ifstream(base + "type"), tp);
        std::getline(std::ifstream(base + "size"), sz);
        j.begin_object();
        j.str_field("level", lv);
        j.str_field("type", tp);
        j.str_field("size", sz);
        j.end_object();
    }
    j.end_array();
}

std::string arg_value(int& i, int argc, char** argv) {
    if (i + 1 >= argc)
        throw std::runtime_error(std::string("missing value after ") + argv[i]);
    return argv[++i];
}

} // namespace

int main(int argc, char** argv) {
    expcode::set_ieee_fp_env();
    std::string cmd = argc > 1 ? argv[1] : "all";
    std::string gpt2_path, qwen_path, out = "results/bench.json";
    int prefill = 32;
    int steps = 8;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gpt2")
            gpt2_path = arg_value(i, argc, argv);
        else if (a == "--qwen")
            qwen_path = arg_value(i, argc, argv);
        else if (a == "--out")
            out = arg_value(i, argc, argv);
        else if (a == "--prefill")
            prefill = std::atoi(arg_value(i, argc, argv).c_str());
        else if (a == "--steps")
            steps = std::atoi(arg_value(i, argc, argv).c_str());
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    const bool do_stream = cmd == "all" || cmd == "stream";
    const bool do_gemv = cmd == "all" || cmd == "gemv";
    const bool do_decode = cmd == "all" || cmd == "decode";
    const expcode::IsaInfo isa = expcode::detect_isa();
    const int ncpu = std::max(1, isa.logical_cpus);
    std::vector<int> threads;
    for (int t = 1; t <= ncpu; ++t)
        threads.push_back(t);
    const expcode::Backend best =
        (isa.avx512f && isa.avx512bw) ? expcode::Backend::Avx512 : expcode::Backend::Avx2;

    Json j;
    j.str_field("gflops_unit", "1e9 binary32 flops, each FMA counts as 2");
    j.str_field("GBps_unit", "1e9 bytes per second");
    j.str_field(
        "accumulation",
        "partial[c & 63] = fma(w[c], x[c], partial[c & 63]); y = partial[0]+...+partial[63]");
    j.begin_object_key("hardware");
    j.str_field("brand", isa.brand);
    j.int_field("family", isa.family);
    j.int_field("model", isa.model);
    j.int_field("stepping", isa.stepping);
    j.int_field("logical_cpus", ncpu);
    j.str_field("cpu_mhz", cpu_mhz());
    j.bool_field("avx2", isa.avx2);
    j.bool_field("avx512f", isa.avx512f);
    j.bool_field("avx512bw", isa.avx512bw);
    j.int_field("mem_total_kib", mem_kib("MemTotal:"));
    j.int_field("mem_available_kib", mem_kib("MemAvailable:"));
#if defined(__clang__)
    j.str_field("compiler", std::string("clang ") + __clang_version__);
#else
    j.str_field("compiler", std::string("gcc ") + __VERSION__);
#endif
    add_caches(j);
    j.end_object();

    j.begin_object_key("peak_fma_gflops");
    if (isa.avx2 && isa.fma) {
        const double a = expcode::peak_fma_gflops(expcode::Backend::Avx2);
        const double b = expcode::peak_fma_gflops(expcode::Backend::Avx2);
        j.num_field("avx2", std::max(a, b));
        std::fprintf(stderr, "peak avx2 %.2f GFLOP/s\n", std::max(a, b));
    }
    if (isa.avx512f && isa.avx512bw) {
        const double a = expcode::peak_fma_gflops(expcode::Backend::Avx512);
        const double b = expcode::peak_fma_gflops(expcode::Backend::Avx512);
        j.num_field("avx512", std::max(a, b));
        std::fprintf(stderr, "peak avx512 %.2f GFLOP/s\n", std::max(a, b));
    }
    j.end_object();

    if (do_stream)
        bench_stream(j, ncpu);

    j.begin_array("gemv");
    auto run_pack = [&](const std::string& path, const std::string& model,
                        const std::vector<std::string>& names, const std::string& tile_name) {
        if (path.empty())
            return;
        auto tensors = load_pack(path);
        std::map<std::string, PackedTensor> idx;
        for (auto& t : tensors) {
            std::string n = t.name;
            idx.emplace(std::move(n), std::move(t));
        }
        tensors.clear();
        for (const std::string& name : names) {
            GemvCase g = take_case(idx, name);
            const bool small = double(g.rows) * g.cols < 3e6;
            const bool f32_ok = double(g.rows) * g.cols * 4.0 < 600e6;
            const bool sweep = (name == tile_name);
            bench_matrix(j, model, g, isa, threads, f32_ok, small, sweep);
            if (name == tile_name) {
                const std::size_t target = 512ull << 20;
                const std::size_t reps = std::max(std::size_t(1), target / (g.bf16.size() * 2));
                if (reps > 1 && mem_kib("MemAvailable:") > 2500000) {
                    GemvCase tiled;
                    tiled.name = g.name + "#tiled";
                    tiled.cols = g.cols;
                    tiled.rows = g.rows * int(reps);
                    tiled.bf16.resize(std::size_t(tiled.rows) * tiled.cols);
                    for (std::size_t r = 0; r < reps; ++r) {
                        std::memcpy(tiled.bf16.data() + r * g.bf16.size(), g.bf16.data(),
                                    g.bf16.size() * sizeof(uint16_t));
                    }
                    g.bf16.clear();
                    g.bf16.shrink_to_fit();
                    bench_matrix(j, model, tiled, isa, threads, false, false, false);
                }
            }
        }
    };
    if (do_gemv) {
        run_pack(gpt2_path, "gpt2",
                 {"h.0.attn.c_attn.weight", "h.0.attn.c_proj.weight", "h.0.mlp.c_fc.weight",
                  "h.0.mlp.c_proj.weight", "wte.weight"},
                 "h.0.mlp.c_proj.weight");
        run_pack(qwen_path, "qwen2.5-0.5b",
                 {"model.layers.0.self_attn.q_proj.weight",
                  "model.layers.0.self_attn.k_proj.weight", "model.layers.0.mlp.gate_proj.weight",
                  "model.layers.0.mlp.down_proj.weight", "model.embed_tokens.weight"},
                 "model.layers.0.mlp.down_proj.weight");
    }
    j.end_array();

    j.begin_array("decode");
    if (do_decode && !gpt2_path.empty()) {
        std::fprintf(stderr, "loading gpt2\n");
        Model model = load_gpt2(gpt2_path);
        bench_decode(j, model, "gpt2", best, threads, prefill, steps);
    }
    if (do_decode && !qwen_path.empty()) {
        std::fprintf(stderr, "loading qwen2.5-0.5b\n");
        Model model = load_qwen2(qwen_path);
        bench_decode(j, model, "qwen2.5-0.5b", best, threads, prefill, steps);
    }
    j.end_array();

    std::ofstream outf(out);
    if (!outf)
        throw std::runtime_error("cannot write " + out);
    outf << j.finish();
    std::fprintf(stderr, "wrote %s\n", out.c_str());
    return 0;
}
