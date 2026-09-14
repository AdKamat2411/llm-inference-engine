#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

constexpr std::size_t MIB = 1024 * 1024;
constexpr int N_LAYERS = 16;
constexpr int KV_SIZE = 512;
constexpr int BLOCK_SIZE = 16;
constexpr int NUM_BLOCKS = 1024;
constexpr int MAX_SEQ_LEN = 2048;

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cerr << cudaGetErrorString(status) << '\n';
        std::exit(1);
    }
}

void measureAllocation(const char *name, std::size_t bytes_per_array) {
    std::size_t free_before, free_after, total;
    check(cudaMemGetInfo(&free_before, &total));
    void *k = nullptr;
    void *v = nullptr;
    check(cudaMalloc(&k, bytes_per_array));
    check(cudaMalloc(&v, bytes_per_array));
    check(cudaMemGetInfo(&free_after, &total));
    std::cout << name << ": requested=" << 2 * bytes_per_array / MIB
              << " MiB, free-memory delta=" << (free_before - free_after) / MIB
              << " MiB\n";
    check(cudaFree(k));
    check(cudaFree(v));
}

int main(int argc, char **argv) {
    constexpr std::size_t bytes_per_token =
        2 * N_LAYERS * KV_SIZE * sizeof(__nv_bfloat16);
    constexpr std::size_t bytes_per_block = BLOCK_SIZE * bytes_per_token;

    measureAllocation("contiguous KV cache", MAX_SEQ_LEN * bytes_per_token / 2);
    measureAllocation("paged KV pool", BLOCK_SIZE * NUM_BLOCKS * bytes_per_token / 2);

    std::vector<int> lengths;
    for (int i = 1; i < argc; ++i) {
        int length = std::atoi(argv[i]);
        if (length < 1 || length > MAX_SEQ_LEN) {
            std::cerr << "Sequence lengths must be in [1, " << MAX_SEQ_LEN << "]\n";
            return 1;
        }
        lengths.push_back(length);
    }
    if (lengths.empty()) {
        lengths.push_back(260);
    }

    int tokens = 0;
    int blocks = 0;
    for (int length : lengths) {
        tokens += length;
        blocks += (length + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }
    if (blocks > NUM_BLOCKS) {
        std::cerr << "Requests exceed the configured pool\n";
        return 1;
    }

    std::cout << "requests=" << lengths.size() << ", cached tokens=" << tokens
              << ", occupied blocks=" << blocks << '/' << NUM_BLOCKS << '\n';
    std::cout << "occupied block bytes=" << blocks * bytes_per_block
              << ", live KV bytes=" << tokens * bytes_per_token
              << ", unused tail bytes=" << (blocks * BLOCK_SIZE - tokens) * bytes_per_token
              << '\n';
    return 0;
}
