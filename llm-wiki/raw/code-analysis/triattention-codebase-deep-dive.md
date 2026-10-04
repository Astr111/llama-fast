# Codebase Extract: TriAttention Implementation in llama.cpp

**Source Path**: `/src/src/llama-triattention.h`, `/src/src/llama-triattention.cpp`, `/src/ggml/src/ggml-cuda/triattention-score.cu`  
**Calibration File**: `/src/bonsai-27b.triattention`  
**Research Paper**: arXiv 2604.04921 (MIT / NVIDIA / ZJU)

## 1. Binary Calibration Header Format (`.triattention`)
```text
magic:          0x54524941 ("TRIA")
version:        1
head_dim:       128
num_layers:     e.g. 36
num_attn_heads: e.g. 36
num_kv_heads:   e.g. 4 (GQA)
rope_theta:     e.g. 10000.0 or 500000.0
rope_style:     0 = half (Llama/Qwen), 1 = interleaved
n_sampled:      sampled (layer, head) pairs
freq_count:     head_dim / 2 (64)
Per head stats:
  q_mean_real[freq_count]  Re(E[q_f])
  q_mean_imag[freq_count]  Im(E[q_f])
  q_abs_mean[freq_count]   E[||q_f||]
  r_f[freq_count]          ||E[q_f]|| / E[||q_f||]
```

## 2. Mathematical Scoring Pipeline

### Step A: Inverse RoPE Rotation
To calculate true interaction probabilities, post-RoPE cached keys must be un-rotated back to base space:
$$\text{out}[f] = \text{in}[f]\cos(\omega_f \cdot \text{pos}) + \text{in}[f+fc]\sin(\omega_f \cdot \text{pos})$$
$$\text{out}[f+fc] = \text{in}[f+fc]\cos(\omega_f \cdot \text{pos}) - \text{in}[f]\sin(\omega_f \cdot \text{pos})$$

### Step B: Trigonometric & MLR Norm Scoring
For each key token $i$ at position $p_k$:
1. $k_f = \text{complex}(\text{pre\_rope\_k}[i \cdot hd + f], \text{pre\_rope\_k}[i \cdot hd + f + fc])$
2. $\text{amp}_f = \|E[q_f]\| \cdot |k_f|$
3. $\phi_f = \text{atan2}(\text{Im}(E[q_f]\bar{k}_f), \text{Re}(E[q_f]\bar{k}_f))$
4. $\text{extra}_f = (E[\|q_f\|] - \|E[q_f]\|) \cdot |k_f|$ (MLR norm excess)
5. Over geometric future offsets $d \in \{1, 2, 4, 8, \dots, 65536\}$:
   $$S_{\text{trig}} = \sum_f \text{amp}_f \cdot \text{scale}_f^2 \cdot \cos(\omega_f(\Delta + d) + \phi_f)$$
   $$S_{\text{norm}} = \sum_f \text{extra}_f \cdot \text{scale}_f^2$$
   $$\text{Score} = \text{aggregate}(S_{\text{trig}} + S_{\text{norm}})$$

## 3. GPU Shared-Memory WHT Inversion
When keys are quantized using TurboQuant (`turbo2` or `turbo3`), the GPU scoring kernel executes cooperative 7-stage Fast Walsh-Hadamard Transform in shared memory (`inverse_wht_rotation_128`) prior to RoPE inversion and scoring.
