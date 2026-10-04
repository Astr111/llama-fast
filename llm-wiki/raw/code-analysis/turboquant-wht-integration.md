# Codebase Extract: TurboQuant KV Cache Quantization & WHT Integration

**Source Path**: `/src/ggml/src/ggml-turbo-quant.c`, `/src/ggml/src/ggml-cuda/turbo-wht.cu`, `/src/src/llama-graph.cpp`, `/src/ggml/include/ggml.h`  
**Paper**: arXiv 2504.19874 (ICLR 2026)

## 1. Supported Quantization Formats in KV Cache

| GGML Type | ID | Structure | Total Bytes / Elements | Bits / Element | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `GGML_TYPE_TURBO2_0` | 45 | `norm (fp16) + qs[QK/4]` | 10 B / 32 elem | 2.5 bpw (effective 2-bit) | 2-bit PolarQuant (no QJL) |
| `GGML_TYPE_TURBO3_0` | 43 | `norm (fp16) + qs[QK/4] + signs[QK/8]` | 14 B / 32 elem | 3.5 bpw (effective 3-bit) | 2-bit PolarQuant + 1-bit WHT signs |
| `GGML_TYPE_TURBO4_0` | 44 | `norm (fp16) + qjl_scale + qs + signs` | 68 B / 128 elem | 4.25 bpw (effective 4-bit) | 3-bit PolarQuant + 1-bit QJL projection |

## 2. Walsh-Hadamard Transform (WHT) Mathematical Formulation
The WHT rotation matrix $R$ is constructed as:
$$R = S_2 \cdot H_d \cdot S_1 \cdot \frac{1}{\sqrt{d}}$$
Where:
- $S_1, S_2 \in \{-1, +1\}^d$ are diagonal random sign matrices (seeded with 42).
- $H_d$ is the Sylvester-Hadamard butterfly recursion matrix ($d=128 \implies 7$ butterfly stages).
- $R$ is strictly orthonormal ($R^T R = I$).

### Inner Product Invariance
Because $R$ is orthonormal:
$$\langle R Q, R K \rangle = (R Q)^T (R K) = Q^T R^T R K = Q^T K = \langle Q, K \rangle$$
Attention logits are mathematically exact up to quantization tolerance!

## 3. Computation Graph Integration (`llama-graph.cpp`)
1. **Query Transformation**:
   Before attention dot product:
   `q = ggml_turbo_wht(ctx0, q, 0, 128, innerq_scale); // direction 0 = forward`
2. **Key Storage**:
   Keys are normalized and rotated by forward WHT upon entry into the KV cache, then quantized into PolarQuant centroids.
3. **Value Storage**:
   Values are rotated by forward WHT and quantized.
4. **Attention Output Un-rotation**:
   After matrix multiplication with rotated values $R V$, the attention output is inverse-transformed:
   `cur = ggml_turbo_wht(ctx0, cur, 1, turbo_group, innerq_scale); // direction 1 = inverse`
