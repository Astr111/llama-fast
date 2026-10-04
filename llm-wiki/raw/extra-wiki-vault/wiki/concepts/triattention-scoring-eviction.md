---
title: "TriAttention Scoring and Eviction"
type: "concept"
tags: ["triattention", "eviction", "kv-cache", "rope-inversion", "trigonometric", "cuda"]
created: 2026-10-04
updated: 2026-10-04
sources: ["[[source-src-triattention]]"]
status: "active"
---

# TriAttention Scoring and Eviction

**TriAttention Scoring and Eviction** is the algorithmic engine within llama.cpp that bounds KV cache memory growth by dynamically evaluating the future importance of cached tokens and evicting redundant keys.

## 1. Why Naive Eviction Fails
- Standard eviction schemes (e.g. H2O, StreamingLLM, or FIFO) evaluate historical attention weights. However, attention scores received in past steps do not accurately predict attention needed at future positions due to Rotary Position Embedding (RoPE) phase rotation.
- RoPE imparts a frequency-dependent phase shift $\omega_f \cdot \Delta$. As relative distance $\Delta = \text{pos}_q - \text{pos}_k$ changes, attention scores oscillate periodically.

## 2. The Trigonometric Formula (arXiv 2604.04921)
TriAttention models future attention importance by calculating expected interaction between calibrated pre-RoPE Query vectors and cached Key vectors across geometric future horizons $d \in \{1, 2, 4, 8, \dots, 65536\}$.

### Step 1: Pre-RoPE Key Inversion
Since cached keys are stored post-RoPE, TriAttention first inverts the rotation:
$$\text{pre\_rope\_k}[f] = \text{post}[f]\cos(\omega_f \cdot p) + \text{post}[f+fc]\sin(\omega_f \cdot p)$$
$$\text{pre\_rope\_k}[f+fc] = \text{post}[f+fc]\cos(\omega_f \cdot p) - \text{post}[f]\sin(\omega_f \cdot p)$$

### Step 2: Complex Amplitude & Phase Extraction
For each frequency band $f \in [0, \text{head\_dim}/2)$:
$$k_f = \text{pre\_rope\_k}[f] + i \cdot \text{pre\_rope\_k}[f+fc]$$
$$\text{amp}_f = \|E[q_f]\| \cdot |k_f|$$
$$\phi_f = \text{atan2}(\text{Im}(E[q_f]\bar{k}_f), \text{Re}(E[q_f]\bar{k}_f))$$

### Step 3: Trigonometric Future Estimation + MLR Norm Term
$$S_{\text{trig}} = \sum_{f} \text{amp}_f \cdot \text{scale}_f^2 \cdot \cos(\omega_f (\Delta + d) + \phi_f)$$
$$S_{\text{norm}} = \sum_{f} (E[\|q_f\|] - \|E[q_f]\|) \cdot |k_f| \cdot \text{scale}_f^2$$
$$\text{Score}(k) = \frac{1}{|D|} \sum_{d \in D} (S_{\text{trig}} + S_{\text{norm}})$$

## 3. Pruning Modes & Granularity
- **`TRIATTENTION_MODE_GLOBAL` (Default)**: Union-based selection across all heads. Each head nominates its top-$B$ candidates, and the global top-$B$ are selected.
- **`TRIATTENTION_MODE_PER_KV_HEAD`**: Independent top-$B$ selection for each Grouped Query Attention (GQA) KV head.
- **`TRIATTENTION_MODE_PER_LAYER_HEAD`**: Independent selection per layer and per KV head.

## 4. Protected Zones
TriAttention guarantees two regions are never pruned:
1. **Initial Attention Sinks**: Prompt prefix tokens (preventing softmax denominator collapse).
2. **Local Sliding Window**: Immediate recent tokens preserving syntax and conversational coherence.

## See Also
- Calibration data format: [[bonsai-model]]
- Rotation interaction: [[walsh-hadamard-transform]]
- Quantization complement: [[turboquant-kv-cache]]
- GPU implementation: [[ggml-cuda-kernels]]
