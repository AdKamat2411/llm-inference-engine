#include "kernels.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

int main() {
    constexpr int heads = 32;
    constexpr int lengths[] = {1, 2, 3, 63, 64, 65, 1023, 1024, 1025, 1537, 2048};

    for (int length : lengths) {
        std::vector<__nv_bfloat16> input(heads * length);
        for (int head = 0; head < heads; ++head) {
            for (int token = 0; token < length; ++token) {
                float value = head == 0 ? 3.0f
                                        : static_cast<float>((token * 37 + head * 11) % 101 - 50) / 8.0f;
                if (head == 1 && token == length / 2) {
                    value += 80.0f;
                }
                input[head * length + token] = (__nv_bfloat16)value;
            }
        }

        __nv_bfloat16 *device_input = nullptr;
        cudaError_t status = cudaMalloc(&device_input, input.size() * sizeof(__nv_bfloat16));
        if (status != cudaSuccess) {
            std::cerr << "cudaMalloc: " << cudaGetErrorString(status) << '\n';
            return 1;
        }
        status = cudaMemcpy(device_input, input.data(), input.size() * sizeof(__nv_bfloat16),
                            cudaMemcpyHostToDevice);
        if (status == cudaSuccess) {
            decodeSoftmax(device_input, length);
            status = cudaDeviceSynchronize();
        }

        std::vector<__nv_bfloat16> output(input.size());
        if (status == cudaSuccess) {
            status = cudaMemcpy(output.data(), device_input, output.size() * sizeof(__nv_bfloat16),
                                cudaMemcpyDeviceToHost);
        }
        cudaFree(device_input);
        if (status != cudaSuccess) {
            std::cerr << "CUDA failure at length " << length << ": "
                      << cudaGetErrorString(status) << '\n';
            return 1;
        }

        for (int head = 0; head < heads; ++head) {
            float max_val = -std::numeric_limits<float>::infinity();
            for (int token = 0; token < length; ++token) {
                max_val = fmaxf(max_val, (float)input[head * length + token]);
            }
            float sum = 0.0f;
            for (int token = 0; token < length; ++token) {
                sum += expf((float)input[head * length + token] - max_val);
            }
            for (int token = 0; token < length; ++token) {
                int index = head * length + token;
                float expected = expf((float)input[index] - max_val) / sum;
                float actual = (float)output[index];
                if (!std::isfinite(actual) ||
                    fabsf(actual - expected) > fmaxf(1.0e-5f, 0.01f * expected)) {
                    std::cerr << "Mismatch at length " << length << ", head " << head
                              << ", token " << token << ": " << actual
                              << " vs " << expected << '\n';
                    return 1;
                }
            }
        }
    }

    std::cout << "decode softmax edge lengths passed\n";
    return 0;
}
