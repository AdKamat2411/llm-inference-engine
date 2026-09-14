#include <cstdint>
#include <cstddef>
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

struct SharedCache {
    __nv_bfloat16 *k; // k_cache[layer][block][token][kv_dim]
    __nv_bfloat16 *v;
    std::vector<int> free_blocks;
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
constexpr int BLOCK_SIZE = 16;
constexpr int NUM_BLOCKS = 1024;
constexpr int MAX_ACTIVE_REQUESTS = 1;
constexpr int MAX_NEW_TOKENS = 100;

enum class InferencePhase { Waiting, Prefill, Decode, Done };

struct Request {
    std::vector<int> input_tokens;
    int *gpu_input_tokens = nullptr;
    __nv_bfloat16 *hidden_state = nullptr;
    int *device_block_table = nullptr;
    std::vector<int> block_table;
    int next_token = 0;
    int cached_tokens = 0;
    InferencePhase inference_phase;
    std::vector<int> generated_tokens;
};


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

void gqa(cublasHandle_t handle, __nv_bfloat16 *q_proj, __nv_bfloat16 *k_proj,
         __nv_bfloat16 *v_proj, __nv_bfloat16 *attn_scores, __nv_bfloat16 *o,
         int query_tokens, int kv_tokens, InferencePhase phase) {
    constexpr int HEAD_DIM = 64;
    constexpr int QUERY_HEADS = HIDDEN_SIZE / HEAD_DIM;
    constexpr int QUERIES_PER_KV_HEAD = HIDDEN_SIZE / KV_SIZE;
    float alpha = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));
    float beta = 0.0f;

    for (int head = 0; head < QUERY_HEADS; ++head) {
        int kv_head = head / QUERIES_PER_KV_HEAD;
        __nv_bfloat16 *q_head = q_proj + head * HEAD_DIM;
        __nv_bfloat16 *k_head = k_proj + kv_head * HEAD_DIM;
        __nv_bfloat16 *score_head = attn_scores + head * query_tokens * kv_tokens;

        // Token-major rows appear as column-major head matrices with full-row strides.
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     kv_tokens, query_tokens, HEAD_DIM, &alpha,
                     k_head, CUDA_R_16BF, KV_SIZE,
                     q_head, CUDA_R_16BF, HIDDEN_SIZE,
                     &beta, score_head, CUDA_R_16BF, kv_tokens,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }

    if (phase == InferencePhase::Prefill) {
        causalMask(attn_scores, query_tokens);
        softmax(attn_scores, query_tokens);
    } else {
        decodeSoftmax(attn_scores, kv_tokens);
    }

    alpha = 1.0f;
    for (int head = 0; head < QUERY_HEADS; ++head) {
        int kv_head = head / QUERIES_PER_KV_HEAD;
        __nv_bfloat16 *score_head = attn_scores + head * query_tokens * kv_tokens;
        __nv_bfloat16 *v_head = v_proj + kv_head * HEAD_DIM;
        __nv_bfloat16 *o_head = o + head * HEAD_DIM;

        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     HEAD_DIM, query_tokens, kv_tokens, &alpha,
                     v_head, CUDA_R_16BF, KV_SIZE,
                     score_head, CUDA_R_16BF, kv_tokens,
                     &beta, o_head, CUDA_R_16BF, HIDDEN_SIZE,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }
}

struct InferenceWorkspace {
    __nv_bfloat16 *post_rms_norm = nullptr;
    __nv_bfloat16 *q_proj = nullptr;
    __nv_bfloat16 *k_proj = nullptr;
    __nv_bfloat16 *v_proj = nullptr;
    __nv_bfloat16 *attn_scores = nullptr;
    __nv_bfloat16 *attn_output = nullptr;
    __nv_bfloat16 *o_proj = nullptr;
    __nv_bfloat16 *gate = nullptr;
    __nv_bfloat16 *up = nullptr;
    __nv_bfloat16 *down = nullptr;
    __nv_bfloat16 *logits = nullptr;

    InferenceWorkspace(int token_capacity, std::size_t score_elements) {
        std::size_t token_bytes = static_cast<std::size_t>(token_capacity) * sizeof(__nv_bfloat16);
        cudaMalloc(&post_rms_norm, token_bytes * HIDDEN_SIZE);
        cudaMalloc(&q_proj, token_bytes * HIDDEN_SIZE);
        cudaMalloc(&k_proj, token_bytes * KV_SIZE);
        cudaMalloc(&v_proj, token_bytes * KV_SIZE);
        cudaMalloc(&attn_scores, score_elements * sizeof(__nv_bfloat16));
        cudaMalloc(&attn_output, token_bytes * HIDDEN_SIZE);
        cudaMalloc(&o_proj, token_bytes * HIDDEN_SIZE);
        cudaMalloc(&gate, token_bytes * INTERMEDIATE_SIZE);
        cudaMalloc(&up, token_bytes * INTERMEDIATE_SIZE);
        cudaMalloc(&down, token_bytes * HIDDEN_SIZE);
        cudaMalloc(&logits, VOCAB_SIZE * sizeof(__nv_bfloat16));
    }

    ~InferenceWorkspace() {
        cudaFree(post_rms_norm);
        cudaFree(q_proj);
        cudaFree(k_proj);
        cudaFree(v_proj);
        cudaFree(attn_scores);
        cudaFree(attn_output);
        cudaFree(o_proj);
        cudaFree(gate);
        cudaFree(up);
        cudaFree(down);
        cudaFree(logits);
    }

    InferenceWorkspace(const InferenceWorkspace &) = delete;
    InferenceWorkspace &operator=(const InferenceWorkspace &) = delete;
};

void runLayer(cublasHandle_t handle, const Weights &weights,
              __nv_bfloat16 *hidden_state, int layer, int query_tokens,
              int position, InferencePhase phase, SharedCache &cache, Request &request,
              InferenceWorkspace &workspace) {
    int logical_block = position / BLOCK_SIZE;
    int slot = position % BLOCK_SIZE;
    int block_num = request.block_table[logical_block];
    int offset = ((layer * NUM_BLOCKS + block_num) * BLOCK_SIZE + slot) * KV_SIZE;

    __nv_bfloat16 *k_write = cache.k + offset;
    __nv_bfloat16 *v_write = cache.v + offset;
    __nv_bfloat16 *k_proj = phase == InferencePhase::Prefill ? workspace.k_proj : k_write;
    __nv_bfloat16 *v_proj = phase == InferencePhase::Prefill ? workspace.v_proj : v_write;

    int kv_tokens = position + query_tokens;

    rmsNorm(workspace.post_rms_norm, hidden_state,
            weights.input_layernorm[layer], query_tokens);
    linear(handle, workspace.post_rms_norm, weights.w_q[layer], workspace.q_proj,
           query_tokens, HIDDEN_SIZE, HIDDEN_SIZE);
    linear(handle, workspace.post_rms_norm, weights.w_k[layer], k_proj,
           query_tokens, HIDDEN_SIZE, KV_SIZE);
    linear(handle, workspace.post_rms_norm, weights.w_v[layer], v_proj,
           query_tokens, HIDDEN_SIZE, KV_SIZE);

    if (phase == InferencePhase::Prefill) {
        ropeV2(workspace.q_proj, HIDDEN_SIZE, query_tokens);
        ropeV2(k_proj, KV_SIZE, query_tokens);
    } else {
        ropeV2Decode(workspace.q_proj, HIDDEN_SIZE, position);
        ropeV2Decode(k_write, KV_SIZE, position);
    }

    if (phase == InferencePhase::Prefill) {
        gqa(handle, workspace.q_proj, k_proj, v_proj, workspace.attn_scores,
            workspace.attn_output, query_tokens, kv_tokens, phase);
    } else {
        pagedAttentionDecode(workspace.q_proj, cache.k, cache.v,
                             workspace.attn_output, request.device_block_table,
                             kv_tokens, layer);
    }


    int p = position;
    if (phase == InferencePhase::Prefill) {
        for (int i = 0; i < query_tokens; i++) {
            p = position + i;
            int dest_block = request.block_table[p / BLOCK_SIZE];
            int dest_slot = p % BLOCK_SIZE;
            int dest_offset = ((layer * NUM_BLOCKS + dest_block) * BLOCK_SIZE + dest_slot) * KV_SIZE;

            __nv_bfloat16 *dest_k = cache.k + dest_offset;
            __nv_bfloat16 *dest_v = cache.v + dest_offset;
            cudaMemcpy(dest_k, workspace.k_proj + i * KV_SIZE, KV_SIZE * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
            cudaMemcpy(dest_v, workspace.v_proj + i * KV_SIZE, KV_SIZE * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
        }
    }

    linear(handle, workspace.attn_output, weights.w_o[layer], workspace.o_proj,
           query_tokens, HIDDEN_SIZE, HIDDEN_SIZE);
    residual(hidden_state, workspace.o_proj, query_tokens);

    rmsNorm(workspace.post_rms_norm, hidden_state,
            weights.post_attention_layernorm[layer], query_tokens);
    linear(handle, workspace.post_rms_norm, weights.gate_proj[layer], workspace.gate,
           query_tokens, HIDDEN_SIZE, INTERMEDIATE_SIZE);
    linear(handle, workspace.post_rms_norm, weights.up_proj[layer], workspace.up,
           query_tokens, HIDDEN_SIZE, INTERMEDIATE_SIZE);
    siluMultiply(workspace.gate, workspace.up, query_tokens);
    linear(handle, workspace.gate, weights.down_proj[layer], workspace.down,
           query_tokens, INTERMEDIATE_SIZE, HIDDEN_SIZE);
    residual(hidden_state, workspace.down, query_tokens);
}

int selectNextToken(cublasHandle_t handle, const Weights &weights,
                    __nv_bfloat16 *hidden_state, int query_tokens,
                    InferenceWorkspace &workspace) {
    rmsNorm(workspace.post_rms_norm, hidden_state, weights.norm, query_tokens);
    linear(handle,
           workspace.post_rms_norm + (query_tokens - 1) * HIDDEN_SIZE,
           weights.embed_tokens, workspace.logits,
           1, HIDDEN_SIZE, VOCAB_SIZE);

    std::vector<__nv_bfloat16> logits_cpu(VOCAB_SIZE);
    cudaMemcpy(logits_cpu.data(), workspace.logits,
               VOCAB_SIZE * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    int next_token = 0;
    float best_logit = -std::numeric_limits<float>::infinity();
    for (int token = 0; token < VOCAB_SIZE; ++token) {
        float value = __bfloat162float(logits_cpu[token]);
        if (value > best_logit) {
            best_logit = value;
            next_token = token;
        }
    }
    return next_token;
}

void prefill(cublasHandle_t handle, const Weights &weights,
             SharedCache &cache, Request &request) {
    int num_tokens = static_cast<int>(request.input_tokens.size());
    InferenceWorkspace workspace(
        MAX_PROMPT_LEN, static_cast<std::size_t>(32) * MAX_PROMPT_LEN * MAX_PROMPT_LEN);
    for (int layer = 0; layer < N_LAYERS; ++layer) {
        runLayer(handle, weights, request.hidden_state, layer, num_tokens, 0,
                 InferencePhase::Prefill, cache, request, workspace);
    }
    request.next_token = selectNextToken(handle, weights, request.hidden_state, num_tokens, workspace);
    request.cached_tokens = num_tokens;
}

void allocateBlocks(SharedCache &cache, Request &request, int blocks_needed) {
    int block_idx;
    for (int i = 0; i < blocks_needed; i++) {
        block_idx = cache.free_blocks.back();
        request.block_table.push_back(block_idx);
        cache.free_blocks.pop_back();
    }
    cudaMemcpy(request.device_block_table, request.block_table.data(),
               request.block_table.size() * sizeof(int), cudaMemcpyHostToDevice);
}

void releaseBlocks(SharedCache &cache, Request &request) {
    for (int block : request.block_table) {
        cache.free_blocks.push_back(block);
    }
    request.block_table.clear();
}

bool setupRequest(SharedCache &cache, const Weights &weights, Request &request) {
    int num_tokens = static_cast<int>(request.input_tokens.size());
    int blocks_needed = (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_tokens == 0 || num_tokens > MAX_PROMPT_LEN ||
        blocks_needed > static_cast<int>(cache.free_blocks.size())) {
        return false;
    }

    if (cudaMalloc(&request.gpu_input_tokens, num_tokens * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&request.hidden_state,
                   static_cast<std::size_t>(MAX_PROMPT_LEN) * HIDDEN_SIZE * sizeof(__nv_bfloat16)) != cudaSuccess ||
        cudaMalloc(&request.device_block_table,
                   ((MAX_SEQ_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE) * sizeof(int)) != cudaSuccess) {
        cudaFree(request.gpu_input_tokens);
        cudaFree(request.hidden_state);
        cudaFree(request.device_block_table);
        request.gpu_input_tokens = nullptr;
        request.hidden_state = nullptr;
        request.device_block_table = nullptr;
        return false;
    }

    cudaMemcpy(request.gpu_input_tokens, request.input_tokens.data(),
               num_tokens * sizeof(int), cudaMemcpyHostToDevice);
    embeddingGather(request.gpu_input_tokens, request.hidden_state,
                    weights.embed_tokens, num_tokens);
    allocateBlocks(cache, request, blocks_needed);
    return true;
}

void releaseRequest(SharedCache &cache, Request &request) {
    releaseBlocks(cache, request);
    cudaFree(request.gpu_input_tokens);
    cudaFree(request.hidden_state);
    cudaFree(request.device_block_table);
    request.gpu_input_tokens = nullptr;
    request.hidden_state = nullptr;
    request.device_block_table = nullptr;
}

void decode(cublasHandle_t handle, const Weights &weights,
            SharedCache &cache, Request &request) {
    int position = request.cached_tokens;
    int logical_block = position / BLOCK_SIZE;
    if (logical_block == static_cast<int>(request.block_table.size())) {
        allocateBlocks(cache, request, 1);
    }

    InferenceWorkspace workspace(1, static_cast<std::size_t>(32) * MAX_SEQ_LEN);
    for (int layer = 0; layer < N_LAYERS; ++layer) {
        runLayer(handle, weights, request.hidden_state, layer, 1, position,
                 InferencePhase::Decode, cache, request, workspace);
    }
    request.next_token = selectNextToken(handle, weights, request.hidden_state, 1, workspace);
    request.cached_tokens++;
}

int main() {
    Weights weights{};
    if (loadWeights(weights) != 0) {
        return 1;
    }

    std::vector<Request> requests;
    std::queue<int> waiting;   // request indices waiting for prefill
    std::vector<int> active;   // admitted request indices

    Request request1{};
    request1.input_tokens = {791, 6864, 315, 9822, 374};
    request1.inference_phase = InferencePhase::Waiting;

    Request request2{};
    request2.input_tokens = {678, 264, 1933, 13};
    request2.inference_phase = InferencePhase::Waiting;

    requests.push_back(request1);
    requests.push_back(request2);
    waiting.push(0);
    waiting.push(1);

    cublasHandle_t handle;
    cublasCreate(&handle);

    SharedCache cache{};
    for (int i = NUM_BLOCKS - 1; i >= 0; i--) {
        cache.free_blocks.push_back(i);
    }
    cudaMalloc(&cache.k, BLOCK_SIZE * NUM_BLOCKS* sizeof(__nv_bfloat16) * KV_SIZE * N_LAYERS);
    cudaMalloc(&cache.v, BLOCK_SIZE * NUM_BLOCKS* sizeof(__nv_bfloat16) * KV_SIZE * N_LAYERS);

    while (!waiting.empty() || !active.empty()) {
        while (active.size() < MAX_ACTIVE_REQUESTS && !waiting.empty()) {
            int request_id = waiting.front();
            Request &front_request = requests[request_id];
            front_request.inference_phase = InferencePhase::Prefill;
            if (!setupRequest(cache, weights, front_request)) {
                std::cerr << "Could not set up request\n";
                cudaFree(cache.k);
                cudaFree(cache.v);
                cublasDestroy(handle);
                return 1;
            }
            active.push_back(request_id);
            waiting.pop();
        }

        std::vector<int> remaining;
        for (int req_id : active) {
            Request &active_req = requests[req_id];
            if (active_req.inference_phase == InferencePhase::Prefill) {
                prefill(handle, weights, cache, active_req);
                active_req.inference_phase = InferencePhase::Decode;
            } else if (active_req.inference_phase == InferencePhase::Decode) {
                cudaMemcpy(active_req.gpu_input_tokens, &active_req.next_token,
                           sizeof(active_req.next_token), cudaMemcpyHostToDevice);
                embeddingGather(active_req.gpu_input_tokens, active_req.hidden_state,
                                weights.embed_tokens, 1);
                decode(handle, weights, cache, active_req);
            }

            if (active_req.next_token != END_OF_SEQ && active_req.cached_tokens < MAX_SEQ_LEN) {
                active_req.generated_tokens.push_back(active_req.next_token);
                std::cout << "Generated token ID: " << active_req.next_token << '\n';
            }

            bool finished = active_req.next_token == END_OF_SEQ ||
                            active_req.cached_tokens >= MAX_SEQ_LEN ||
                            active_req.generated_tokens.size() >= MAX_NEW_TOKENS;
            if (finished) {
                releaseRequest(cache, active_req);
                active_req.inference_phase = InferencePhase::Done;
            } else {
                remaining.push_back(req_id);
            }
        }
        active.swap(remaining);
    }

    cudaFree(cache.k);
    cudaFree(cache.v);
    cublasDestroy(handle);
    return 0;
}
