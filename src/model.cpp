#include "model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>

namespace {

std::map<std::string, PackedTensor> index_pack(std::vector<PackedTensor> tensors) {
    std::map<std::string, PackedTensor> out;
    for (auto& t : tensors) {
        std::string name = t.name;
        out.emplace(std::move(name), std::move(t));
    }
    return out;
}

PackedTensor take(std::map<std::string, PackedTensor>& idx, const std::string& name) {
    auto it = idx.find(name);
    if (it == idx.end())
        throw std::runtime_error("missing tensor " + name);
    PackedTensor t = std::move(it->second);
    idx.erase(it);
    return t;
}

bool has(const std::map<std::string, PackedTensor>& idx, const std::string& name) {
    return idx.find(name) != idx.end();
}

std::vector<float> as_fp32_1d(const PackedTensor& t, int n, const std::string& name) {
    if (t.dims.size() != 1 || int(t.dims[0]) != n || int(t.bf16.size()) != n) {
        throw std::runtime_error("bad 1d tensor " + name);
    }
    std::vector<float> out(t.bf16.size());
    expcode::promote_bf16(t.bf16.data(), out.data(), t.bf16.size());
    return out;
}

void assign_2d(Linear& linear, PackedTensor t, int rows, int cols) {
    if (t.dims.size() != 2 || int(t.dims[0]) != rows || int(t.dims[1]) != cols ||
        int(t.bf16.size()) != rows * cols) {
        throw std::runtime_error("bad 2d tensor " + t.name);
    }
    linear.assign(t.name, std::move(t.bf16), rows, cols);
}

} // namespace

void Linear::assign(std::string name_in, std::vector<uint16_t> weight, int rows_in, int cols_in) {
    name = std::move(name_in);
    rows = rows_in;
    cols = cols_in;
    bf16 = std::move(weight);
    encoded = expcode::EncodedMatrix::encode(bf16, uint32_t(rows), uint32_t(cols), 0);
}

void Linear::assign_bias(const std::vector<uint16_t>& bias_bf16) {
    if (int(bias_bf16.size()) != rows) {
        throw std::runtime_error("bias length does not match rows of " + name);
    }
    bias.resize(bias_bf16.size());
    expcode::promote_bf16(bias_bf16.data(), bias.data(), bias_bf16.size());
}

Model load_gpt2(const std::string& pack_path) {
    Model m;
    m.arch = Model::Arch::Gpt2;
    m.n_layer = 12;
    m.n_embd = 768;
    m.n_head = 12;
    m.n_kv_head = 12;
    m.head_dim = 64;
    m.intermediate = 3072;
    m.n_vocab = 50257;
    m.kv_dim = m.n_embd;
    m.eps = 1e-5f;
    auto idx = index_pack(load_pack(pack_path));
    {
        PackedTensor wte = take(idx, "wte.weight");
        assign_2d(m.lm_head, std::move(wte), m.n_vocab, m.n_embd);
    }
    {
        PackedTensor wpe = take(idx, "wpe.weight");
        if (wpe.dims.size() != 2 || int(wpe.dims[1]) != m.n_embd) {
            throw std::runtime_error("bad wpe");
        }
        m.wpe.resize(wpe.bf16.size());
        expcode::promote_bf16(wpe.bf16.data(), m.wpe.data(), wpe.bf16.size());
    }
    m.final_norm_w = as_fp32_1d(take(idx, "ln_f.weight"), m.n_embd, "ln_f.weight");
    m.final_norm_b = as_fp32_1d(take(idx, "ln_f.bias"), m.n_embd, "ln_f.bias");
    m.gpt2.resize(m.n_layer);
    for (int i = 0; i < m.n_layer; ++i) {
        const std::string p = "h." + std::to_string(i);
        Gpt2Layer& layer = m.gpt2[i];
        assign_2d(layer.c_attn, take(idx, p + ".attn.c_attn.weight"), 3 * m.n_embd, m.n_embd);
        layer.c_attn.assign_bias(take(idx, p + ".attn.c_attn.bias").bf16);
        assign_2d(layer.c_proj, take(idx, p + ".attn.c_proj.weight"), m.n_embd, m.n_embd);
        layer.c_proj.assign_bias(take(idx, p + ".attn.c_proj.bias").bf16);
        assign_2d(layer.c_fc, take(idx, p + ".mlp.c_fc.weight"), m.intermediate, m.n_embd);
        layer.c_fc.assign_bias(take(idx, p + ".mlp.c_fc.bias").bf16);
        assign_2d(layer.mlp_proj, take(idx, p + ".mlp.c_proj.weight"), m.n_embd, m.intermediate);
        layer.mlp_proj.assign_bias(take(idx, p + ".mlp.c_proj.bias").bf16);
        layer.ln1_w = as_fp32_1d(take(idx, p + ".ln_1.weight"), m.n_embd, p + ".ln_1.weight");
        layer.ln1_b = as_fp32_1d(take(idx, p + ".ln_1.bias"), m.n_embd, p + ".ln_1.bias");
        layer.ln2_w = as_fp32_1d(take(idx, p + ".ln_2.weight"), m.n_embd, p + ".ln_2.weight");
        layer.ln2_b = as_fp32_1d(take(idx, p + ".ln_2.bias"), m.n_embd, p + ".ln_2.bias");
    }
    return m;
}

Model load_qwen2(const std::string& pack_path) {
    Model m;
    m.arch = Model::Arch::Qwen2;
    m.n_layer = 24;
    m.n_embd = 896;
    m.n_head = 14;
    m.n_kv_head = 2;
    m.head_dim = 64;
    m.intermediate = 4864;
    m.n_vocab = 151936;
    m.kv_dim = m.n_kv_head * m.head_dim;
    m.eps = 1e-6f;
    m.rope_theta = 1000000.f;
    m.rope_inv_freq.resize(m.head_dim / 2);
    for (int i = 0; i < m.head_dim / 2; ++i) {
        m.rope_inv_freq[i] =
            std::exp(-std::log(m.rope_theta) * (2.f * float(i)) / float(m.head_dim));
    }
    auto idx = index_pack(load_pack(pack_path));
    assign_2d(m.lm_head, take(idx, "model.embed_tokens.weight"), m.n_vocab, m.n_embd);
    m.final_norm_w = as_fp32_1d(take(idx, "model.norm.weight"), m.n_embd, "model.norm.weight");
    m.qwen.resize(m.n_layer);
    for (int i = 0; i < m.n_layer; ++i) {
        const std::string p = "model.layers." + std::to_string(i);
        QwenLayer& layer = m.qwen[i];
        assign_2d(layer.q, take(idx, p + ".self_attn.q_proj.weight"), m.n_embd, m.n_embd);
        assign_2d(layer.k, take(idx, p + ".self_attn.k_proj.weight"), m.kv_dim, m.n_embd);
        assign_2d(layer.v, take(idx, p + ".self_attn.v_proj.weight"), m.kv_dim, m.n_embd);
        assign_2d(layer.o, take(idx, p + ".self_attn.o_proj.weight"), m.n_embd, m.n_embd);
        layer.q.assign_bias(take(idx, p + ".self_attn.q_proj.bias").bf16);
        layer.k.assign_bias(take(idx, p + ".self_attn.k_proj.bias").bf16);
        layer.v.assign_bias(take(idx, p + ".self_attn.v_proj.bias").bf16);
        if (has(idx, p + ".self_attn.o_proj.bias")) {
            layer.o.assign_bias(take(idx, p + ".self_attn.o_proj.bias").bf16);
        }
        assign_2d(layer.gate, take(idx, p + ".mlp.gate_proj.weight"), m.intermediate, m.n_embd);
        assign_2d(layer.up, take(idx, p + ".mlp.up_proj.weight"), m.intermediate, m.n_embd);
        assign_2d(layer.down, take(idx, p + ".mlp.down_proj.weight"), m.n_embd, m.intermediate);
        layer.rms1 = as_fp32_1d(take(idx, p + ".input_layernorm.weight"), m.n_embd, p);
        layer.rms2 = as_fp32_1d(take(idx, p + ".post_attention_layernorm.weight"), m.n_embd, p);
    }
    return m;
}

DecodeWorkspace make_workspace(const Model& model, int max_pos) {
    DecodeWorkspace ws;
    ws.max_pos = max_pos;
    const int e = model.n_embd;
    const int inter = model.intermediate;
    const int kv = model.kv_dim;
    ws.h.resize(e);
    ws.norm.resize(e);
    ws.q.resize(model.arch == Model::Arch::Gpt2 ? 3 * e : e);
    ws.k.resize(kv);
    ws.v.resize(kv);
    ws.attn.resize(e);
    ws.proj.resize(e);
    ws.hidden.resize(inter);
    ws.gate.resize(inter);
    ws.up.resize(inter);
    ws.logits.resize(model.n_vocab);
    ws.score.resize(max_pos);
    ws.kcache.assign(std::size_t(model.n_layer) * max_pos * kv, 0.f);
    ws.vcache.assign(std::size_t(model.n_layer) * max_pos * kv, 0.f);
    return ws;
}

namespace {

void layernorm(const float* x, const float* w, const float* b, float* y, int n, float eps) {
    float mean = 0.f;
    for (int i = 0; i < n; ++i)
        mean += x[i];
    mean /= float(n);
    float var = 0.f;
    for (int i = 0; i < n; ++i) {
        const float d = x[i] - mean;
        var += d * d;
    }
    var /= float(n);
    const float inv = 1.f / std::sqrt(var + eps);
    for (int i = 0; i < n; ++i)
        y[i] = (x[i] - mean) * inv * w[i] + b[i];
}

void rmsnorm(const float* x, const float* w, float* y, int n, float eps) {
    float ss = 0.f;
    for (int i = 0; i < n; ++i)
        ss += x[i] * x[i];
    const float inv = 1.f / std::sqrt(ss / float(n) + eps);
    for (int i = 0; i < n; ++i)
        y[i] = x[i] * inv * w[i];
}

float gelu_new(float x) {
    constexpr float k0 = 0.7978845608028654f; // sqrt(2/pi)
    constexpr float k1 = 0.044715f;
    const float u = k0 * (x + k1 * x * x * x);
    return 0.5f * x * (1.f + std::tanh(u));
}

float silu(float x) {
    return x / (1.f + std::exp(-x));
}

void softmax(float* s, int n) {
    float m = s[0];
    for (int i = 1; i < n; ++i)
        m = std::max(m, s[i]);
    float sum = 0.f;
    for (int i = 0; i < n; ++i) {
        s[i] = std::exp(s[i] - m);
        sum += s[i];
    }
    const float inv = 1.f / sum;
    for (int i = 0; i < n; ++i)
        s[i] *= inv;
}

double apply_linear(const Linear& linear, const float* x, float* y, expcode::Backend backend,
                    int threads, bool use_encoded) {
    const auto t0 = std::chrono::steady_clock::now();
    if (use_encoded) {
        expcode::gemv_encoded(linear.encoded.view(), x, y, backend, threads);
    } else {
        expcode::gemv_bf16(linear.bf16.data(), x, y, linear.rows, linear.cols, backend, threads);
    }
    for (int i = 0; i < int(linear.bias.size()); ++i)
        y[i] += linear.bias[i];
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

float* cache_at(std::vector<float>& cache, int layer, int pos, int max_pos, int kv_dim) {
    return cache.data() + (std::size_t(layer) * max_pos + pos) * kv_dim;
}

void attend(const float* q, const float* kcache, const float* vcache, float* attn, float* score,
            int pos, int n_head, int n_kv_head, int head_dim, int kv_dim) {
    const float scale = 1.f / std::sqrt(float(head_dim));
    const int groups = n_head / n_kv_head;
    const int tcount = pos + 1;
    for (int h = 0; h < n_head; ++h) {
        const int kvh = h / groups;
        const float* qh = q + h * head_dim;
        for (int t = 0; t < tcount; ++t) {
            const float* kh = kcache + std::size_t(t) * kv_dim + kvh * head_dim;
            float dot = 0.f;
            for (int i = 0; i < head_dim; ++i)
                dot += qh[i] * kh[i];
            score[t] = dot * scale;
        }
        softmax(score, tcount);
        float* oh = attn + h * head_dim;
        for (int i = 0; i < head_dim; ++i) {
            float acc = 0.f;
            for (int t = 0; t < tcount; ++t) {
                const float* vh = vcache + std::size_t(t) * kv_dim + kvh * head_dim;
                acc += score[t] * vh[i];
            }
            oh[i] = acc;
        }
    }
}

void rope(float* heads, int n_heads, int head_dim, int pos, const float* inv_freq) {
    for (int h = 0; h < n_heads; ++h) {
        float* p = heads + h * head_dim;
        for (int i = 0; i < head_dim / 2; ++i) {
            const float ang = float(pos) * inv_freq[i];
            const float c = std::cos(ang);
            const float s = std::sin(ang);
            const float a = p[i];
            const float b = p[i + head_dim / 2];
            p[i] = a * c - b * s;
            p[i + head_dim / 2] = a * s + b * c;
        }
    }
}

StepStats step_gpt2(const Model& model, DecodeWorkspace& ws, int token, int pos,
                    expcode::Backend backend, int threads, bool use_encoded) {
    StepStats stats;
    const int e = model.n_embd;
    if (std::size_t(pos + 1) * std::size_t(e) > model.wpe.size()) {
        throw std::runtime_error("position exceeds wpe");
    }
    const uint16_t* row = model.lm_head.bf16.data() + std::size_t(token) * e;
    for (int i = 0; i < e; ++i) {
        ws.h[i] = expcode::bf16_to_fp32(row[i]) + model.wpe[std::size_t(pos) * e + i];
    }
    for (int layer = 0; layer < model.n_layer; ++layer) {
        const Gpt2Layer& L = model.gpt2[layer];
        layernorm(ws.h.data(), L.ln1_w.data(), L.ln1_b.data(), ws.norm.data(), e, model.eps);
        stats.gemv_seconds +=
            apply_linear(L.c_attn, ws.norm.data(), ws.q.data(), backend, threads, use_encoded);
        float* kc = cache_at(ws.kcache, layer, 0, ws.max_pos, model.kv_dim);
        float* vc = cache_at(ws.vcache, layer, 0, ws.max_pos, model.kv_dim);
        float* krow = kc + std::size_t(pos) * model.kv_dim;
        float* vrow = vc + std::size_t(pos) * model.kv_dim;
        for (int i = 0; i < e; ++i) {
            krow[i] = ws.q[e + i];
            vrow[i] = ws.q[2 * e + i];
        }
        attend(ws.q.data(), kc, vc, ws.attn.data(), ws.score.data(), pos, model.n_head,
               model.n_kv_head, model.head_dim, model.kv_dim);
        stats.gemv_seconds +=
            apply_linear(L.c_proj, ws.attn.data(), ws.proj.data(), backend, threads, use_encoded);
        for (int i = 0; i < e; ++i)
            ws.h[i] += ws.proj[i];
        layernorm(ws.h.data(), L.ln2_w.data(), L.ln2_b.data(), ws.norm.data(), e, model.eps);
        stats.gemv_seconds +=
            apply_linear(L.c_fc, ws.norm.data(), ws.hidden.data(), backend, threads, use_encoded);
        for (int i = 0; i < model.intermediate; ++i)
            ws.hidden[i] = gelu_new(ws.hidden[i]);
        stats.gemv_seconds += apply_linear(L.mlp_proj, ws.hidden.data(), ws.proj.data(), backend,
                                           threads, use_encoded);
        for (int i = 0; i < e; ++i)
            ws.h[i] += ws.proj[i];
    }
    layernorm(ws.h.data(), model.final_norm_w.data(), model.final_norm_b.data(), ws.norm.data(), e,
              model.eps);
    stats.gemv_seconds += apply_linear(model.lm_head, ws.norm.data(), ws.logits.data(), backend,
                                       threads, use_encoded);
    stats.logits = ws.logits;
    return stats;
}

StepStats step_qwen(const Model& model, DecodeWorkspace& ws, int token, int pos,
                    expcode::Backend backend, int threads, bool use_encoded) {
    StepStats stats;
    const int e = model.n_embd;
    const uint16_t* row = model.lm_head.bf16.data() + std::size_t(token) * e;
    for (int i = 0; i < e; ++i)
        ws.h[i] = expcode::bf16_to_fp32(row[i]);
    for (int layer = 0; layer < model.n_layer; ++layer) {
        const QwenLayer& L = model.qwen[layer];
        rmsnorm(ws.h.data(), L.rms1.data(), ws.norm.data(), e, model.eps);
        stats.gemv_seconds +=
            apply_linear(L.q, ws.norm.data(), ws.q.data(), backend, threads, use_encoded);
        stats.gemv_seconds +=
            apply_linear(L.k, ws.norm.data(), ws.k.data(), backend, threads, use_encoded);
        stats.gemv_seconds +=
            apply_linear(L.v, ws.norm.data(), ws.v.data(), backend, threads, use_encoded);
        rope(ws.q.data(), model.n_head, model.head_dim, pos, model.rope_inv_freq.data());
        rope(ws.k.data(), model.n_kv_head, model.head_dim, pos, model.rope_inv_freq.data());
        float* kc = cache_at(ws.kcache, layer, 0, ws.max_pos, model.kv_dim);
        float* vc = cache_at(ws.vcache, layer, 0, ws.max_pos, model.kv_dim);
        float* krow = kc + std::size_t(pos) * model.kv_dim;
        float* vrow = vc + std::size_t(pos) * model.kv_dim;
        for (int i = 0; i < model.kv_dim; ++i) {
            krow[i] = ws.k[i];
            vrow[i] = ws.v[i];
        }
        attend(ws.q.data(), kc, vc, ws.attn.data(), ws.score.data(), pos, model.n_head,
               model.n_kv_head, model.head_dim, model.kv_dim);
        stats.gemv_seconds +=
            apply_linear(L.o, ws.attn.data(), ws.proj.data(), backend, threads, use_encoded);
        for (int i = 0; i < e; ++i)
            ws.h[i] += ws.proj[i];
        rmsnorm(ws.h.data(), L.rms2.data(), ws.norm.data(), e, model.eps);
        stats.gemv_seconds +=
            apply_linear(L.gate, ws.norm.data(), ws.gate.data(), backend, threads, use_encoded);
        stats.gemv_seconds +=
            apply_linear(L.up, ws.norm.data(), ws.up.data(), backend, threads, use_encoded);
        for (int i = 0; i < model.intermediate; ++i) {
            ws.hidden[i] = silu(ws.gate[i]) * ws.up[i];
        }
        stats.gemv_seconds +=
            apply_linear(L.down, ws.hidden.data(), ws.proj.data(), backend, threads, use_encoded);
        for (int i = 0; i < e; ++i)
            ws.h[i] += ws.proj[i];
    }
    rmsnorm(ws.h.data(), model.final_norm_w.data(), ws.norm.data(), e, model.eps);
    stats.gemv_seconds += apply_linear(model.lm_head, ws.norm.data(), ws.logits.data(), backend,
                                       threads, use_encoded);
    stats.logits = ws.logits;
    return stats;
}

} // namespace

StepStats decode_step(const Model& model, DecodeWorkspace& ws, int token, int pos,
                      expcode::Backend backend, int threads, bool use_encoded) {
    if (pos < 0 || pos >= ws.max_pos)
        throw std::runtime_error("decode position out of range");
    if (token < 0 || token >= model.n_vocab)
        throw std::runtime_error("token out of range");
    if (model.arch == Model::Arch::Gpt2) {
        return step_gpt2(model, ws, token, pos, backend, threads, use_encoded);
    }
    return step_qwen(model, ws, token, pos, backend, threads, use_encoded);
}
