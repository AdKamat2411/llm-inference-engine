#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <queue>
#include "kernels.cuh"

#define JSON_USE_IMPLICIT_CONVERSIONS 0 // this one is useful, but I don't remember why
#include "json.hpp"

using json = nlohmann::json;

constexpr uint64_t B_TO_MB = 1024 * 1024;
constexpr uint64_t B_TO_GB = 1024ull * 1024 * 1024;

int checkGPUStatus() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0)
    {
        std::cerr << "No CUDA devices found\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Global memory: " << prop.totalGlobalMem / B_TO_MB << " MB\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    size_t free_mem;
    size_t total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Free memory: " << free_mem / B_TO_GB << "GB, total memory: " << total_mem / B_TO_GB << "GB\n";
    return 0;
}

struct Weights {
    __nv_bfloat16 *embed_tokens;
    __nv_bfloat16 *input_layernorm[16];
    __nv_bfloat16 *gate_proj[16];
    __nv_bfloat16 *up_proj[16];
    __nv_bfloat16 *down_proj[16];
    __nv_bfloat16 *w_q[16];
    __nv_bfloat16 *w_k[16];
    __nv_bfloat16 *w_v[16];
    __nv_bfloat16 *w_o[16];
    __nv_bfloat16 *post_attention_layernorm[16];
    __nv_bfloat16 *norm;
};

struct KVCache {
    __nv_bfloat16 *k_cache;
    __nv_bfloat16 *v_cache;
};

int loadWeights(Weights &weights) {
    if (checkGPUStatus() != 0) {
        return 1;
    }

    std::ifstream file("model.safetensors", std::ios_base::binary);
    if (!file.is_open()) {
        std::cerr << "Can't open model.safetensors\n";
        return 1;
    }

    uint64_t header_size;
    file.read(reinterpret_cast<char *>(&header_size), 8);

    std::string header(header_size, '\0');
    file.read(header.data(), header_size);
    json header_json = json::parse(header);

    std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> offsets;
    uint64_t max_offset = 0;

    for (auto& [tensor_name, tensor_data] : header_json.items()) {
        if (tensor_name == "__metadata__") { continue; }

        std::string dtype = tensor_data["dtype"].get<std::string>();
        auto& shape = tensor_data["shape"];
        uint64_t start = tensor_data["data_offsets"][0].get<uint64_t>();
        uint64_t end = tensor_data["data_offsets"][1].get<uint64_t>();

        if (end > max_offset) {
            max_offset = end;
        }

        offsets[tensor_name] = {start, end};
    }
    uint64_t data_size = max_offset;
    std::vector<char> buf(data_size);
    file.read(buf.data(), static_cast<std::streamsize>(data_size));

    void* gpu_weights;
    cudaMalloc(&gpu_weights, data_size);
    cudaMemcpy(gpu_weights, buf.data(), data_size, cudaMemcpyHostToDevice);

    auto gpu_ptr = [&](const std::string& name) {
        return reinterpret_cast<__nv_bfloat16*>(
            static_cast<char*>(gpu_weights) + offsets.at(name).first);
    };

    weights.embed_tokens = gpu_ptr("model.embed_tokens.weight");
    weights.norm = gpu_ptr("model.norm.weight");
    for (int i = 0; i < 16; ++i) {
        std::string layer = "model.layers." + std::to_string(i);
        weights.input_layernorm[i] = gpu_ptr(layer + ".input_layernorm.weight");
        weights.gate_proj[i] = gpu_ptr(layer + ".mlp.gate_proj.weight");
        weights.up_proj[i] = gpu_ptr(layer + ".mlp.up_proj.weight");
        weights.down_proj[i] = gpu_ptr(layer + ".mlp.down_proj.weight");
        weights.w_q[i] = gpu_ptr(layer + ".self_attn.q_proj.weight");
        weights.w_k[i] = gpu_ptr(layer + ".self_attn.k_proj.weight");
        weights.w_v[i] = gpu_ptr(layer + ".self_attn.v_proj.weight");
        weights.w_o[i] = gpu_ptr(layer + ".self_attn.o_proj.weight");
        weights.post_attention_layernorm[i] = gpu_ptr(layer + ".post_attention_layernorm.weight");
    }

    return 0;
}

constexpr int MAX_PROMPT_LEN = 512;
constexpr int MAX_SEQ_LEN = 2048;
constexpr int N_LAYERS = 16;
constexpr int HIDDEN_SIZE = 2048;
constexpr int KV_SIZE = 512;
constexpr int INTERMEDIATE_SIZE = 8192;
constexpr int VOCAB_SIZE = 128256;
constexpr int END_OF_SEQ = 128009;

void linear(cublasHandle_t handle, __nv_bfloat16 *input, __nv_bfloat16 *weight,
            __nv_bfloat16 *output, int num_tokens, int input_size, int output_size) {
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        output_size,
        num_tokens,
        input_size,
        &alpha,
        weight,
        CUDA_R_16BF,
        input_size,
        input,
        CUDA_R_16BF,
        input_size,
        &beta,
        output,
        CUDA_R_16BF,
        output_size,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );
}

void gqa(cublasHandle_t handle, __nv_bfloat16 *q_proj, __nv_bfloat16 *k_proj, __nv_bfloat16 *v_proj, __nv_bfloat16 *attn_scores, __nv_bfloat16 *o, int num_tokens) {
    float alpha = 1.0f / sqrtf(64.0f);
    float beta = 0.0f;

    for (int i = 0; i < 32; i++) {
        int k_head_idx = i / 4;
        __nv_bfloat16 *q_head = q_proj + i * 64;
        __nv_bfloat16 *k_head = k_proj + k_head_idx * 64;
        __nv_bfloat16 *attn_score_head = attn_scores + i * num_tokens * num_tokens;

        cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            num_tokens,
            num_tokens,
            64,
            &alpha,
            k_head,
            CUDA_R_16BF,
            512,
            q_head,
            CUDA_R_16BF,
            2048,
            &beta,
            attn_score_head,
            CUDA_R_16BF,
            num_tokens,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        );
    }

    causalMask(attn_scores, num_tokens);
    softmax(attn_scores, num_tokens);

    alpha = 1.0f;
    for (int i = 0; i < 32; i++) {
        int v_head_idx = i / 4;
        __nv_bfloat16 *attn_score_head = attn_scores + i * num_tokens * num_tokens;
        __nv_bfloat16 *v_head = v_proj + v_head_idx * 64;
        __nv_bfloat16 *o_head = o + i * 64;

        cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            64,
            num_tokens,
            num_tokens,
            &alpha,
            v_head,
            CUDA_R_16BF,
            512,
            attn_score_head,
            CUDA_R_16BF,
            num_tokens,
            &beta,
            o_head,
            CUDA_R_16BF,
            2048,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        );
    }
}

void decode_gqa(cublasHandle_t handle, __nv_bfloat16 *q_proj, __nv_bfloat16 *k_proj, __nv_bfloat16 *v_proj, __nv_bfloat16 *attn_scores, __nv_bfloat16 *o, int L) {
    float alpha = 1.0f / sqrtf(64.0f);
    float beta = 0.0f;

    for (int i = 0; i < 32; i++) {
        int k_head_idx = i / 4;
        __nv_bfloat16 *q_head = q_proj + i * 64;
        __nv_bfloat16 *k_head = k_proj + k_head_idx * 64;
        __nv_bfloat16 *attn_score_head = attn_scores + i * L;

        cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            L,
            1,
            64,
            &alpha,
            k_head,
            CUDA_R_16BF,
            512,
            q_head,
            CUDA_R_16BF,
            2048,
            &beta,
            attn_score_head,
            CUDA_R_16BF,
            L,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        );
    }

    decodeSoftmax(attn_scores, L);

    alpha = 1.0f;
    for (int i = 0; i < 32; i++) {
        int v_head_idx = i / 4;
        __nv_bfloat16 *attn_score_head = attn_scores + i * L;
        __nv_bfloat16 *v_head = v_proj + v_head_idx * 64;
        __nv_bfloat16 *o_head = o + i * 64;

        cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            64,
            1,
            L,
            &alpha,
            v_head,
            CUDA_R_16BF,
            512,
            attn_score_head,
            CUDA_R_16BF,
            L,
            &beta,
            o_head,
            CUDA_R_16BF,
            2048,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        );
    }
}

int prefill(cublasHandle_t handle, const Weights &weights, __nv_bfloat16 *hidden_state, int num_tokens, KVCache &kv_cache) {
    __nv_bfloat16 *post_rms_norm;
    __nv_bfloat16 *q_proj;
    __nv_bfloat16 *k_proj;
    __nv_bfloat16 *v_proj;
    __nv_bfloat16 *attn_scores;
    __nv_bfloat16 *attn_output;
    __nv_bfloat16 *o_proj;
    __nv_bfloat16 *gate;
    __nv_bfloat16 *up;
    __nv_bfloat16 *down;
    __nv_bfloat16 *logits;


    cudaMalloc(&post_rms_norm, MAX_PROMPT_LEN * sizeof(__nv_bfloat16) * HIDDEN_SIZE);
    cudaMalloc(&q_proj, MAX_PROMPT_LEN * sizeof(__nv_bfloat16) * HIDDEN_SIZE);
    cudaMalloc(&attn_scores, 32 * MAX_PROMPT_LEN * MAX_PROMPT_LEN * sizeof(__nv_bfloat16));
    cudaMalloc(&attn_output, MAX_PROMPT_LEN * sizeof(__nv_bfloat16) * HIDDEN_SIZE);
    cudaMalloc(&o_proj, MAX_PROMPT_LEN * sizeof(__nv_bfloat16) * HIDDEN_SIZE);
    cudaMalloc(&gate, MAX_PROMPT_LEN * sizeof(__nv_bfloat16) * INTERMEDIATE_SIZE);
    cudaMalloc(&up, MAX_PROMPT_LEN * sizeof(__nv_bfloat16) * INTERMEDIATE_SIZE);
    cudaMalloc(&down, MAX_PROMPT_LEN * sizeof(__nv_bfloat16) * HIDDEN_SIZE);
    cudaMalloc(&logits, VOCAB_SIZE * sizeof(__nv_bfloat16));

    for (int layer = 0; layer < N_LAYERS; ++layer) {
        k_proj = kv_cache.k_cache + MAX_SEQ_LEN * KV_SIZE * layer;
        v_proj = kv_cache.v_cache + MAX_SEQ_LEN * KV_SIZE * layer;

        rmsNorm(post_rms_norm, hidden_state, weights.input_layernorm[layer], num_tokens);

        linear(handle, post_rms_norm, weights.w_q[layer], q_proj,
               num_tokens, HIDDEN_SIZE, HIDDEN_SIZE);

        linear(handle, post_rms_norm, weights.w_k[layer], k_proj,
               num_tokens, HIDDEN_SIZE, KV_SIZE);
        linear(handle, post_rms_norm, weights.w_v[layer], v_proj,
               num_tokens, HIDDEN_SIZE, KV_SIZE);

        ropeV2(q_proj, HIDDEN_SIZE, num_tokens);
        ropeV2(k_proj, KV_SIZE, num_tokens);
        gqa(handle, q_proj, k_proj, v_proj, attn_scores, attn_output, num_tokens);

        linear(handle, attn_output, weights.w_o[layer], o_proj,
               num_tokens, HIDDEN_SIZE, HIDDEN_SIZE);
        residual(hidden_state, o_proj, num_tokens);

        rmsNorm(post_rms_norm, hidden_state,
                weights.post_attention_layernorm[layer], num_tokens);
        linear(handle, post_rms_norm, weights.gate_proj[layer], gate,
               num_tokens, HIDDEN_SIZE, INTERMEDIATE_SIZE);
        linear(handle, post_rms_norm, weights.up_proj[layer], up,
               num_tokens, HIDDEN_SIZE, INTERMEDIATE_SIZE);
        siluMultiply(gate, up, num_tokens);
        linear(handle, gate, weights.down_proj[layer], down,
               num_tokens, INTERMEDIATE_SIZE, HIDDEN_SIZE);
        residual(hidden_state, down, num_tokens);
    }

    rmsNorm(post_rms_norm, hidden_state, weights.norm, num_tokens);
    linear(handle,
           post_rms_norm + (num_tokens - 1) * HIDDEN_SIZE,
           weights.embed_tokens,
           logits,
           1,
           HIDDEN_SIZE,
           VOCAB_SIZE);

    std::vector<__nv_bfloat16> logits_cpu(VOCAB_SIZE);
    cudaMemcpy(logits_cpu.data(), logits,
               VOCAB_SIZE * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);

    int next_token = 0;
    float best_logit = -std::numeric_limits<float>::infinity();
    for (int token = 0; token < VOCAB_SIZE; ++token) {
        float value = __bfloat162float(logits_cpu[token]);
        if (value > best_logit) {
            best_logit = value;
            next_token = token;
        }
    }

    cudaFree(post_rms_norm);
    cudaFree(q_proj);
    cudaFree(attn_scores);
    cudaFree(attn_output);
    cudaFree(o_proj);
    cudaFree(gate);
    cudaFree(up);
    cudaFree(down);
    cudaFree(logits);

    return next_token;
}

int decode(cublasHandle_t handle, const Weights &weights, __nv_bfloat16 *hidden_state, int num_tokens, KVCache &kv_cache) {
    int L = num_tokens + 1;
    __nv_bfloat16 *k_curr;
    __nv_bfloat16 *v_curr;

    __nv_bfloat16 *post_rms_norm;
    __nv_bfloat16 *q_proj;
    __nv_bfloat16 *attn_scores;
    __nv_bfloat16 *attn_output;
    __nv_bfloat16 *o_proj;
    __nv_bfloat16 *gate;
    __nv_bfloat16 *up;
    __nv_bfloat16 *down;
    __nv_bfloat16 *logits;

    cudaMalloc(&post_rms_norm, sizeof(__nv_bfloat16) * HIDDEN_SIZE);
    cudaMalloc(&q_proj, sizeof(__nv_bfloat16) * HIDDEN_SIZE);
    cudaMalloc(&attn_scores, 32 * MAX_SEQ_LEN * sizeof(__nv_bfloat16));
    cudaMalloc(&attn_output, sizeof(__nv_bfloat16) * HIDDEN_SIZE);
    cudaMalloc(&o_proj, sizeof(__nv_bfloat16) * HIDDEN_SIZE);
    cudaMalloc(&gate, sizeof(__nv_bfloat16) * INTERMEDIATE_SIZE);
    cudaMalloc(&up, sizeof(__nv_bfloat16) * INTERMEDIATE_SIZE);
    cudaMalloc(&down, sizeof(__nv_bfloat16) * HIDDEN_SIZE);
    cudaMalloc(&logits, VOCAB_SIZE * sizeof(__nv_bfloat16));

    for (int layer = 0; layer < N_LAYERS; ++layer) {
        __nv_bfloat16 *k_layer = kv_cache.k_cache + MAX_SEQ_LEN * KV_SIZE * layer;
        __nv_bfloat16 *v_layer = kv_cache.v_cache + MAX_SEQ_LEN * KV_SIZE * layer;
        k_curr = k_layer + num_tokens * KV_SIZE;
        v_curr = v_layer + num_tokens * KV_SIZE;

        rmsNorm(post_rms_norm, hidden_state, weights.input_layernorm[layer], 1);

        linear(handle, post_rms_norm, weights.w_q[layer], q_proj,
               1, HIDDEN_SIZE, HIDDEN_SIZE);

        linear(handle, post_rms_norm, weights.w_k[layer], k_curr,
               1, HIDDEN_SIZE, KV_SIZE);
        linear(handle, post_rms_norm, weights.w_v[layer], v_curr,
               1, HIDDEN_SIZE, KV_SIZE);

        ropeV2Decode(q_proj, HIDDEN_SIZE, num_tokens);
        ropeV2Decode(k_curr, KV_SIZE, num_tokens);
        decode_gqa(handle, q_proj, k_layer, v_layer, attn_scores, attn_output, L);

        linear(handle, attn_output, weights.w_o[layer], o_proj,
               1, HIDDEN_SIZE, HIDDEN_SIZE);
        residual(hidden_state, o_proj, 1);

        rmsNorm(post_rms_norm, hidden_state,
                weights.post_attention_layernorm[layer], 1);
        linear(handle, post_rms_norm, weights.gate_proj[layer], gate,
               1, HIDDEN_SIZE, INTERMEDIATE_SIZE);
        linear(handle, post_rms_norm, weights.up_proj[layer], up,
               1, HIDDEN_SIZE, INTERMEDIATE_SIZE);
        siluMultiply(gate, up, 1);
        linear(handle, gate, weights.down_proj[layer], down,
               1, INTERMEDIATE_SIZE, HIDDEN_SIZE);
        residual(hidden_state, down, 1);
    }

    rmsNorm(post_rms_norm, hidden_state, weights.norm, 1);
    linear(handle,
           post_rms_norm,
           weights.embed_tokens,
           logits,
           1,
           HIDDEN_SIZE,
           VOCAB_SIZE);

    std::vector<__nv_bfloat16> logits_cpu(VOCAB_SIZE);
    cudaMemcpy(logits_cpu.data(), logits,
               VOCAB_SIZE * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);

    int next_token = 0;
    float best_logit = -std::numeric_limits<float>::infinity();
    for (int token = 0; token < VOCAB_SIZE; ++token) {
        float value = __bfloat162float(logits_cpu[token]);
        if (value > best_logit) {
            best_logit = value;
            next_token = token;
        }
    }

    cudaFree(post_rms_norm);
    cudaFree(q_proj);
    cudaFree(attn_scores);
    cudaFree(attn_output);
    cudaFree(o_proj);
    cudaFree(gate);
    cudaFree(up);
    cudaFree(down);
    cudaFree(logits);

    return next_token;
}

int main() {
    Weights weights{};
    if (loadWeights(weights) != 0) {
        return 1;
    }

    std::vector<int> input_tokens = {791, 6864, 315, 9822, 374};
    cublasHandle_t handle;
    cublasCreate(&handle);

    int *gpu_input_tokens;

    cudaMalloc(&gpu_input_tokens, MAX_PROMPT_LEN * sizeof(int));
    cudaMemcpy(gpu_input_tokens, input_tokens.data(), input_tokens.size() * sizeof(int), cudaMemcpyHostToDevice);

    __nv_bfloat16 *input_embeddings;
    cudaMalloc(&input_embeddings, MAX_PROMPT_LEN * sizeof(__nv_bfloat16) * 2048);
    embeddingGather(gpu_input_tokens, input_embeddings, weights.embed_tokens, input_tokens.size());

    // this loop output is next loop's input
    __nv_bfloat16 *hidden_state;
    cudaMalloc(&hidden_state, MAX_PROMPT_LEN * sizeof(__nv_bfloat16) * HIDDEN_SIZE);
    int num_tokens = static_cast<int>(input_tokens.size());
    cudaMemcpy(hidden_state, input_embeddings,
               num_tokens * HIDDEN_SIZE * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToDevice);

    KVCache kv_cache{};
    cudaMalloc(&kv_cache.k_cache, MAX_SEQ_LEN * sizeof(__nv_bfloat16) * KV_SIZE * N_LAYERS);
    cudaMalloc(&kv_cache.v_cache, MAX_SEQ_LEN * sizeof(__nv_bfloat16) * KV_SIZE * N_LAYERS);

    int next_token = prefill(handle, weights, hidden_state, num_tokens, kv_cache);
    while (next_token != END_OF_SEQ) {
        std::cout << "Generated token ID: " << next_token << '\n';
        cudaMemcpy(gpu_input_tokens, &next_token, sizeof(next_token), cudaMemcpyHostToDevice);
        embeddingGather(gpu_input_tokens, hidden_state, weights.embed_tokens, 1);
        next_token = decode(handle, weights, hidden_state, num_tokens, kv_cache);
        num_tokens++;
    }

    cudaFree(gpu_input_tokens);
    cudaFree(input_embeddings);
    cudaFree(hidden_state);
    cudaFree(kv_cache.k_cache);
    cudaFree(kv_cache.v_cache);
    cublasDestroy(handle);
    return 0;
}
