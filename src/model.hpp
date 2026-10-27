#pragma once

#include "expcode/gemv.hpp"
#include "pack.hpp"

#include <string>
#include <vector>

struct Linear {
    std::string name;
    int rows = 0;
    int cols = 0;
    std::vector<uint16_t> bf16;
    std::vector<float> bias;
    expcode::EncodedMatrix encoded;

    void assign(std::string name_in, std::vector<uint16_t> weight, int rows_in, int cols_in);
    void assign_bias(const std::vector<uint16_t>& bias_bf16);
};

struct Gpt2Layer {
    Linear c_attn;
    Linear c_proj;
    Linear c_fc;
    Linear mlp_proj;
    std::vector<float> ln1_w, ln1_b, ln2_w, ln2_b;
};

struct QwenLayer {
    Linear q, k, v, o;
    Linear gate, up, down;
    std::vector<float> rms1, rms2;
};

struct Model {
    enum class Arch { Gpt2, Qwen2 };
    Arch arch = Arch::Gpt2;
    int n_layer = 0;
    int n_embd = 0;
    int n_head = 0;
    int n_kv_head = 0;
    int head_dim = 0;
    int intermediate = 0;
    int n_vocab = 0;
    int kv_dim = 0;
    float eps = 1e-5f;
    float rope_theta = 1000000.f;
    Linear lm_head;
    std::vector<float> wpe;
    std::vector<float> final_norm_w, final_norm_b;
    std::vector<float> rope_inv_freq;
    std::vector<Gpt2Layer> gpt2;
    std::vector<QwenLayer> qwen;
};

struct DecodeWorkspace {
    int max_pos = 0;
    std::vector<float> h, norm, q, k, v, attn, proj, hidden, gate, up, logits, score;
    std::vector<float> kcache, vcache;
};

struct StepStats {
    double gemv_seconds = 0;
    std::vector<float> logits;
};

Model load_gpt2(const std::string& pack_path);
Model load_qwen2(const std::string& pack_path);

DecodeWorkspace make_workspace(const Model& model, int max_pos);

// One decode step at `pos`, writing the new K/V into the cache.
// use_encoded selects the expcode GEMV. Otherwise the kernel reads BF16
// weights. Both use the canonical stripe accumulation, so a matching
// backend and any thread count produce identical logits.
StepStats decode_step(const Model& model, DecodeWorkspace& ws, int token, int pos,
                      expcode::Backend backend, int threads, bool use_encoded);
