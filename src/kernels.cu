#include "kernels.cuh"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

// Prefill kernels live here (embedding gather, RMSNorm, RoPE, ...).
// Decode / paged attention come later. Write them as you hit each README section.

__global__ void embeddingGatherKernel(int *gpu_input_tokens, __nv_bfloat16 *input_embeddings, __nv_bfloat16 *embed_tokens) {
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    input_embeddings[workIndex] = embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x];
    input_embeddings[workIndex + 1024] = embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x + 1024];
}

void embeddingGather(int *gpu_input_tokens, __nv_bfloat16 *gpu_input_embeds, __nv_bfloat16 *embed_tokens, int num_input_tokens) {
    embeddingGatherKernel<<<num_input_tokens, 1024>>>(gpu_input_tokens, gpu_input_embeds, embed_tokens);
}

__global__ void rmsNormKernel(__nv_bfloat16 *output, __nv_bfloat16 *input, __nv_bfloat16 *rms_weights) {
    __shared__ float rms_vector[1024];
    int workIdx = threadIdx.x + blockIdx.x * 2048;
    rms_vector[threadIdx.x] = (float)input[workIdx] * (float)input[workIdx] +
                                (float)input[workIdx + 1024] * (float)input[workIdx + 1024];

    __syncthreads();

    for (int i = 1; i < 1024; i *= 2) {
        if (threadIdx.x % (2 * i) == 0) {
            rms_vector[threadIdx.x] += rms_vector[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        rms_vector[0] = sqrt(rms_vector[0] / 2048 + 1.0e-5);
    }
    __syncthreads();

    output[workIdx] = (__nv_bfloat16)((float)input[workIdx] / rms_vector[0] * (float)rms_weights[threadIdx.x]);
    workIdx += 1024;
    output[workIdx] = (__nv_bfloat16)((float)input[workIdx] / rms_vector[0] * (float)rms_weights[threadIdx.x + 1024]);
}

void rmsNorm(__nv_bfloat16 *output, __nv_bfloat16 *input, __nv_bfloat16 *rms_weights, int num_input_tokens) {
    rmsNormKernel<<<num_input_tokens, 1024>>>(output, input, rms_weights);
}


__global__ void ropeKernel(__nv_bfloat16 *input, int proj_dim) {
    float theta = 1.0f / powf(500000.0f, (2.0f * (threadIdx.x % 32)) / 64.0f);
    float angle = blockIdx.x * theta;

    int idx = 2 * threadIdx.x + blockIdx.x * proj_dim;

    float a = (float)input[idx];
    float b = (float)input[idx + 1];

    float new_a = a * cosf(angle) - b * sinf(angle);
    float new_b = a * sinf(angle) + b * cosf(angle);

    input[idx] = (__nv_bfloat16)new_a;
    input[idx + 1] = (__nv_bfloat16)new_b;
}

void rope(__nv_bfloat16 *input, int proj_dim, int num_tokens) {
    ropeKernel<<<num_tokens, proj_dim / 2>>>(input, proj_dim);
}

__global__ void ropeV2Kernel(__nv_bfloat16 *input, int proj_dim, int position_offset) {
    constexpr int head_dim = 64;
    constexpr int half_head_dim = head_dim / 2;
    constexpr float rope_theta = 500000.0f;
    constexpr float scale_factor = 32.0f;
    constexpr float low_freq_factor = 1.0f;
    constexpr float high_freq_factor = 4.0f;
    constexpr float original_context_length = 8192.0f;
    constexpr float two_pi = 6.28318530717958647692f;

    int head_idx = threadIdx.x / half_head_dim;
    int pair_idx = threadIdx.x % half_head_dim;

    float inv_freq = 1.0f / powf(
        rope_theta,
        (2.0f * pair_idx) / head_dim
    );
    float wavelength = two_pi / inv_freq;
    float low_freq_wavelength = original_context_length / low_freq_factor;
    float high_freq_wavelength = original_context_length / high_freq_factor;

    if (wavelength > low_freq_wavelength) {
        inv_freq /= scale_factor;
    } else if (wavelength >= high_freq_wavelength) {
        float smooth = (original_context_length / wavelength - low_freq_factor) /
                       (high_freq_factor - low_freq_factor);
        inv_freq = (1.0f - smooth) * (inv_freq / scale_factor) +
                   smooth * inv_freq;
    }

    int head_start = blockIdx.x * proj_dim + head_idx * head_dim;
    int first_idx = head_start + pair_idx;
    int second_idx = first_idx + half_head_dim;

    float angle = (position_offset + blockIdx.x) * inv_freq;
    float cosine = cosf(angle);
    float sine = sinf(angle);
    float first = (float)input[first_idx];
    float second = (float)input[second_idx];

    input[first_idx] = (__nv_bfloat16)(first * cosine - second * sine);
    input[second_idx] = (__nv_bfloat16)(first * sine + second * cosine);
}

void ropeV2(__nv_bfloat16 *input, int proj_dim, int num_tokens) {
    ropeV2Kernel<<<num_tokens, proj_dim / 2>>>(input, proj_dim, 0);
}

void ropeV2Decode(__nv_bfloat16 *input, int proj_dim, int position) {
    ropeV2Kernel<<<1, proj_dim / 2>>>(input, proj_dim, position);
}


__global__ void residualKernel(__nv_bfloat16 *input, __nv_bfloat16 *input_embeds) {
    int workIdx = threadIdx.x + blockIdx.x * 2048;
    input[workIdx] += input_embeds[workIdx];
    input[workIdx + 1024] += input_embeds[workIdx + 1024];
}

void residual(__nv_bfloat16 *input, __nv_bfloat16 *input_embeds, int num_tokens) {
    residualKernel<<<num_tokens, 1024>>>(input, input_embeds);
}

__global__ void softmaxKernel(__nv_bfloat16 *input, int num_tokens) {
    __shared__ float row[1024];
    __shared__ float max_val;

    int workIdx = threadIdx.x + blockIdx.x * num_tokens;
    float token = (float)input[workIdx];
    row[threadIdx.x] = token;
    __syncthreads();

    for (int i = 1; i < num_tokens; i *= 2) {
        if ((threadIdx.x % (2 * i) == 0) && (threadIdx.x + i < num_tokens)) {
            row[threadIdx.x] = fmaxf(row[threadIdx.x], row[threadIdx.x + i]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        max_val = row[0];
    }
    __syncthreads();

    row[threadIdx.x] = expf(token - max_val);
    __syncthreads();

    for (int i = 1; i < num_tokens; i *= 2) {
        if ((threadIdx.x % (2 * i) == 0) && (threadIdx.x + i < num_tokens)) {
            row[threadIdx.x] += row[threadIdx.x + i];
        }
        __syncthreads();
    }

    input[workIdx] = (__nv_bfloat16)(expf(token - max_val) / row[0]);
}

void softmax(__nv_bfloat16 *input, int num_tokens) {
    softmaxKernel<<<num_tokens * 32, num_tokens>>>(input, num_tokens);
}

void decodeSoftmax(__nv_bfloat16 *input, int seq_len) {
    if (seq_len > 1024) {
        std::cerr << "Decode softmax supports at most 1024 tokens; got " << seq_len << '\n';
        std::exit(EXIT_FAILURE);
    }
    softmaxKernel<<<32, seq_len>>>(input, seq_len);
}

__global__ void causalMaskKernel(__nv_bfloat16 *input, int num_tokens) {
    int workIdx = threadIdx.x + blockIdx.x * num_tokens;
    int token_num = blockIdx.x % num_tokens;

    if (threadIdx.x > token_num) {
        input[workIdx] = (__nv_bfloat16)(-INFINITY);
    }
}

void causalMask(__nv_bfloat16 *input, int num_tokens) {
    causalMaskKernel<<<num_tokens * 32, num_tokens>>>(input, num_tokens);
}

__global__ void siluMultiplyKernel(__nv_bfloat16 *gate, __nv_bfloat16 *up) {
    int workIdx = threadIdx.x + blockIdx.x * 8192;

    for (int offset = 0; offset < 8192; offset += 1024) {
        int idx = workIdx + offset;
        float gate_value = (float)gate[idx];
        float up_value = (float)up[idx];
        float silu_value = gate_value / (1.0f + expf(-gate_value));
        gate[idx] = (__nv_bfloat16)(silu_value * up_value);
    }
}

void siluMultiply(__nv_bfloat16 *gate, __nv_bfloat16 *up, int num_tokens) {
    siluMultiplyKernel<<<num_tokens, 1024>>>(gate, up);
}
