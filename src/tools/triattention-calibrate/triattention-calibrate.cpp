#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#define TRIATTENTION_MAGIC   0x54524941u
#define TRIATTENTION_VERSION 1u

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

struct head_acc {
    std::vector<double> sum_real;
    std::vector<double> sum_imag;
    std::vector<double> sum_abs;
    int64_t count = 0;
};

class triattention_collector {
public:
    int64_t expected_n_head   = 0;
    int64_t detected_head_dim = 0;

    bool collect(struct ggml_tensor * t, bool ask);
    void reset();

    // Map: layer_idx -> vector of head_acc (size n_head)
    std::map<int32_t, std::vector<head_acc>> layer_stats;

private:
    std::mutex m_mutex;
    std::vector<uint8_t> m_host_buf;
    std::vector<float>   m_f32_buf;
};

static bool parse_qcur_layer(const char * name, int32_t & il, bool & is_normed) {
    if (!name) return false;
    is_normed = false;
    // Prefer explicit pre-RoPE normalized tensor for Qwen/Bonsai architectures (TA-11)
    const char * p = strstr(name, "Qcur_normed-");
    if (p) {
        is_normed = true;
        p += 12;
    } else {
        p = strstr(name, "Qcur-");
        if (!p) return false;
        p += 5;
    }
    char * end = nullptr;
    long val = strtol(p, &end, 10);
    if (end == p) return false;
    il = (int32_t)val;
    return true;
}

bool triattention_collector::collect(struct ggml_tensor * t, bool ask) {
    int32_t il = -1;
    bool is_normed = false;
    if (!parse_qcur_layer(t->name, il, is_normed)) {
        return false;
    }

    // Reject post-RoPE tensors (TA-11: TriAttention requires pre-RoPE Q centers)
    if (t->op == GGML_OP_ROPE) {
        return false;
    }

    // Only collect reshaped 3D Q tensor before RoPE [n_embd_head, n_head, n_tokens]
    if (expected_n_head > 0 && t->ne[1] != expected_n_head) {
        return false;
    }
    if (t->ne[0] <= 0 || t->ne[0] % 2 != 0) {
        return false;
    }

    if (ask) {
        return true;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    detected_head_dim = t->ne[0];

    GGML_ASSERT(t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16);
    GGML_ASSERT(ggml_is_contiguous(t));

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    const uint8_t * data;
    if (is_host) {
        data = (const uint8_t *) t->data;
    } else {
        m_host_buf.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, m_host_buf.data(), 0, ggml_nbytes(t));
        data = m_host_buf.data();
    }

    const int64_t head_dim = t->ne[0];
    const int64_t n_head   = t->ne[1];
    const int64_t n_tokens = t->ne[2];
    const int64_t n_elem   = head_dim * n_head * n_tokens;
    const int64_t fc       = head_dim / 2;

    const float * f;
    if (t->type == GGML_TYPE_F32) {
        f = (const float *) data;
    } else {
        m_f32_buf.resize(n_elem);
        if (t->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) data, m_f32_buf.data(), n_elem);
        } else {
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) data, m_f32_buf.data(), n_elem);
        }
        f = m_f32_buf.data();
    }

    auto & heads = layer_stats[il];
    if (heads.empty()) {
        heads.resize(n_head);
        for (int64_t h = 0; h < n_head; ++h) {
            heads[h].sum_real.assign(fc, 0.0);
            heads[h].sum_imag.assign(fc, 0.0);
            heads[h].sum_abs.assign(fc, 0.0);
            heads[h].count = 0;
        }
    }

    // Accumulate frequency-domain statistics (Half style: real in first fc, imag in second fc)
    for (int64_t tok = 0; tok < n_tokens; ++tok) {
        for (int64_t h = 0; h < n_head; ++h) {
            const float * q_ptr = f + (tok * n_head + h) * head_dim;
            auto & h_acc = heads[h];
            for (int64_t k = 0; k < fc; ++k) {
                const float re = q_ptr[k];
                const float im = q_ptr[k + fc];
                const float mag = sqrtf(re * re + im * im);
                h_acc.sum_real[k] += (double)re;
                h_acc.sum_imag[k] += (double)im;
                h_acc.sum_abs[k]  += (double)mag;
            }
            h_acc.count += 1;
        }
    }

    return true;
}

void triattention_collector::reset() {
    std::lock_guard<std::mutex> lock(m_mutex);
    layer_stats.clear();
}

static triattention_collector g_collector;

static bool triattention_calibrate_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);
    return g_collector.collect(t, ask);
}

static void print_usage(int, char ** argv) {
    LOG("\nTriAttention Calibration Tool for llama.cpp\n");
    LOG("Usage:\n");
    LOG("  %s -m model.gguf -f corpus.txt -o model.triattention [-c 2048] [-ngl 28] [-t 6]\n\n", argv[0]);
    LOG("Arguments:\n");
    LOG("  -m, --model PATH              path to GGUF model\n");
    LOG("  -f, --file PATH               path to plain-text calibration corpus\n");
    LOG("  -o, --output PATH             output path for .triattention calibration file\n");
    LOG("  -c, --ctx-size N              context size (default: 2048)\n");
    LOG("  -b, --batch-size N            batch size (default: 512)\n");
    LOG("  -ngl, --n-gpu-layers N        number of GPU layers to offload\n");
    LOG("  -t, --threads N               number of CPU threads\n\n");
}

int main(int argc, char ** argv) {
    common_params params;

    params.out_file = "model.triattention";
    params.n_ctx    = 2048;
    params.n_batch  = 512;
    params.escape   = false;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CLI, print_usage)) {
        return 1;
    }

    if (params.prompt.empty()) {
        LOG_ERR("%s: no calibration text provided (use -f FNAME)\n", __func__);
        return 1;
    }

    if (params.model.path.empty()) {
        LOG_ERR("%s: no model provided (use -m FNAME)\n", __func__);
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = triattention_calibrate_cb_eval;
    params.cb_eval_user_data = nullptr;
    params.warmup            = false;

    LOG_INF("%s: loading model %s ...\n", __func__, params.model.path.c_str());
    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to initialize model or context\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    // Retrieve architecture parameters
    const uint32_t num_layers     = (uint32_t)llama_model_n_layer(model);
    const uint32_t num_attn_heads = (uint32_t)llama_model_n_head(model);
    const uint32_t num_kv_heads   = (uint32_t)llama_model_n_head_kv(model);
    const uint32_t n_embd         = (uint32_t)llama_model_n_embd(model);
    uint32_t head_dim             = (num_attn_heads > 0) ? (n_embd / num_attn_heads) : 128;

    double rope_theta = (double)llama_model_rope_freq_base_train(model);
    if (rope_theta <= 0.0) {
        char rope_buf[64] = {0};
        char arch_buf[64] = {0};
        char arch_key[128] = {0};
        if (llama_model_meta_val_str(model, "general.architecture", arch_buf, sizeof(arch_buf)) > 0) {
            snprintf(arch_key, sizeof(arch_key), "%s.rope.freq_base", arch_buf);
        }
        if ((arch_key[0] != '\0' && llama_model_meta_val_str(model, arch_key, rope_buf, sizeof(rope_buf)) > 0) ||
            llama_model_meta_val_str(model, "qwen35.rope.freq_base", rope_buf, sizeof(rope_buf)) > 0 ||
            llama_model_meta_val_str(model, "qwen3.rope.freq_base",  rope_buf, sizeof(rope_buf)) > 0 ||
            llama_model_meta_val_str(model, "rope.freq_base",        rope_buf, sizeof(rope_buf)) > 0) {
            double parsed = atof(rope_buf);
            if (parsed > 0.0) {
                rope_theta = parsed;
            }
        }
    }
    if (rope_theta <= 0.0) {
        rope_theta = 10000.0;
    }

    g_collector.expected_n_head   = num_attn_heads;

    LOG_INF("%s: model info: layers=%u, attn_heads=%u, kv_heads=%u, rope_theta=%.1f\n",
            __func__, num_layers, num_attn_heads, num_kv_heads, rope_theta);

    LOG_INF("%s: tokenizing calibration text ...\n", __func__);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, params.parse_special);
    LOG_INF("%s: calibration text tokenized to %zu tokens\n", __func__, tokens.size());

    const int32_t n_ctx   = params.n_ctx;
    const int32_t n_batch = std::min(params.n_batch, n_ctx);

    if ((int32_t) tokens.size() < 128) {
        LOG_WRN("%s: calibration text is very short (%zu tokens), recommend >= 1024 tokens\n",
                __func__, tokens.size());
    }

    const int n_tokens_total = (int) tokens.size();
    const int n_chunk = std::max(1, (n_tokens_total + n_ctx - 1) / n_ctx);

    LOG_INF("%s: collecting pre-RoPE Q statistics over %d chunk(s) (context size: %d, batch size: %d)...\n",
            __func__, n_chunk, n_ctx, n_batch);

    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    for (int i = 0; i < n_chunk; ++i) {
        const int start = i * n_ctx;
        const int count = std::min(n_ctx, n_tokens_total - start);
        if (count <= 0) break;

        llama_memory_clear(llama_get_memory(ctx), true);

        for (int j = 0; j < count; j += n_batch) {
            const int n_tok = std::min(n_batch, count - j);

            common_batch_clear(batch);
            for (int k = 0; k < n_tok; ++k) {
                common_batch_add(batch, tokens[start + j + k], j + k, { 0 }, false);
            }

            if (llama_decode(ctx, batch)) {
                LOG_ERR("%s: failed to decode batch in chunk %d\n", __func__, i);
                llama_batch_free(batch);
                return 1;
            }
        }
        LOG_INF("%s: completed chunk %d / %d\n", __func__, i + 1, n_chunk);
    }

    llama_batch_free(batch);

    head_dim = (g_collector.detected_head_dim > 0) ? (uint32_t)g_collector.detected_head_dim : 256;
    const uint32_t freq_count = head_dim / 2;
    std::vector<std::pair<uint32_t, uint32_t>> sampled_pairs;

    for (const auto & kv : g_collector.layer_stats) {
        const uint32_t il = (uint32_t)kv.first;
        for (uint32_t h = 0; h < num_attn_heads; ++h) {
            if (h < kv.second.size() && kv.second[h].count > 0) {
                sampled_pairs.emplace_back(il, h);
            }
        }
    }

    if (sampled_pairs.empty()) {
        LOG_ERR("%s: no attention Q heads were captured during evaluation!\n", __func__);
        return 1;
    }

    LOG_INF("%s: writing TriAttention calibration file to: %s\n", __func__, params.out_file.c_str());
    LOG_INF("%s: captured %zu (layer, head) pairs across %zu attention layers\n",
            __func__, sampled_pairs.size(), g_collector.layer_stats.size());

    FILE * f = fopen(params.out_file.c_str(), "wb");
    if (!f) {
        LOG_ERR("%s: failed to open output file: %s\n", __func__, params.out_file.c_str());
        return 1;
    }

    const uint32_t magic       = TRIATTENTION_MAGIC;   // 0x54524941 ("TRIA")
    const uint32_t version     = TRIATTENTION_VERSION; // 1
    const uint32_t rope_style  = 0;                    // 0 = half
    const uint32_t n_sampled   = (uint32_t)sampled_pairs.size();
    char model_name_buf[128] = {0};
    if (llama_model_meta_val_str(model, "general.name", model_name_buf, sizeof(model_name_buf)) <= 0) {
        llama_model_desc(model, model_name_buf, sizeof(model_name_buf));
    }
    if (model_name_buf[0] == '\0') {
        strncpy(model_name_buf, "Bonsai-2-27B-PQ2_0", sizeof(model_name_buf) - 1);
    }
    const char *   model_name  = model_name_buf;
    const uint32_t name_len    = (uint32_t)strlen(model_name) + 1;

    fwrite(&magic,          sizeof(uint32_t), 1, f);
    fwrite(&version,        sizeof(uint32_t), 1, f);
    fwrite(&head_dim,       sizeof(uint32_t), 1, f);
    fwrite(&num_layers,     sizeof(uint32_t), 1, f);
    fwrite(&num_attn_heads, sizeof(uint32_t), 1, f);
    fwrite(&num_kv_heads,   sizeof(uint32_t), 1, f);
    fwrite(&rope_theta,     sizeof(double),   1, f);
    fwrite(&rope_style,     sizeof(uint32_t), 1, f);
    fwrite(&n_sampled,      sizeof(uint32_t), 1, f);
    fwrite(&freq_count,     sizeof(uint32_t), 1, f);
    fwrite(&name_len,       sizeof(uint32_t), 1, f);
    fwrite(model_name,      1, name_len, f);

    std::vector<float> q_mean_real(freq_count);
    std::vector<float> q_mean_imag(freq_count);
    std::vector<float> q_abs_mean(freq_count);
    std::vector<float> r_f(freq_count);

    for (const auto & p : sampled_pairs) {
        const uint32_t il = p.first;
        const uint32_t h  = p.second;
        const auto & h_acc = g_collector.layer_stats[il][h];
        const double count = (double)std::max((int64_t)1, h_acc.count);

        for (uint32_t k = 0; k < freq_count; ++k) {
            const float re  = (float)(h_acc.sum_real[k] / count);
            const float im  = (float)(h_acc.sum_imag[k] / count);
            const float abs = (float)(h_acc.sum_abs[k]  / count);
            const float mag = sqrtf(re * re + im * im);
            const float r   = mag / std::max(abs, 1e-8f);

            q_mean_real[k] = re;
            q_mean_imag[k] = im;
            q_abs_mean[k]  = abs;
            r_f[k]         = r;
        }

        fwrite(&il, sizeof(uint32_t), 1, f);
        fwrite(&h,  sizeof(uint32_t), 1, f);
        fwrite(q_mean_real.data(), sizeof(float), freq_count, f);
        fwrite(q_mean_imag.data(), sizeof(float), freq_count, f);
        fwrite(q_abs_mean.data(),  sizeof(float), freq_count, f);
        fwrite(r_f.data(),         sizeof(float), freq_count, f);
    }

    fclose(f);

    LOG_INF("%s: Successfully generated %s (size: %.2f KB)\n",
            __func__, params.out_file.c_str(),
            (double)(sizeof(uint32_t)*11 + sizeof(double) + name_len + n_sampled*(sizeof(uint32_t)*2 + sizeof(float)*freq_count*4)) / 1024.0);

    return 0;
}
