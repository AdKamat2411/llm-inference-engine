#pragma once
#include <cuda_bf16.h>

// Declare kernels/wrappers here as you implement them in kernels.cu.
void embeddingGather(int *gpu_input_tokens, __nv_bfloat16 *gpu_input_embeds, __nv_bfloat16 *embed_tokens, int num_input_tokens);
void rmsNorm(__nv_bfloat16 *output, __nv_bfloat16 *input, __nv_bfloat16 *rms_weights, int num_input_tokens);
void rope(__nv_bfloat16 *input, int proj_dim, int num_tokens);
void ropeV2(__nv_bfloat16 *input, int proj_dim, int num_tokens);
void ropeV2Decode(__nv_bfloat16 *input, int proj_dim, int position);
void residual(__nv_bfloat16 *input, __nv_bfloat16 *input_embeds, int num_tokens);
void softmax(__nv_bfloat16 *input, int num_tokens);
void decodeSoftmax(__nv_bfloat16 *input, int seq_len);
void causalMask(__nv_bfloat16 *input, int num_tokens);
void siluMultiply(__nv_bfloat16 *gate, __nv_bfloat16 *up, int num_tokens);
