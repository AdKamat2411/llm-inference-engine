# LLM Inference Engine

A small Llama 3.2 1B inference engine written in C++ and CUDA. The project
implements the model execution path directly, using custom CUDA kernels for
transformer operations and cuBLAS for matrix multiplication.

This is an educational engine for learning model bring-up, GPU kernels, KV-cache
management, and inference benchmarking. It is not a drop-in replacement for
vLLM and does not provide an HTTP API.

## What is implemented

- Direct loading of BF16 weights from a single `model.safetensors` file
- Llama 3.2 1B prefill and autoregressive decode
- Grouped-query attention (32 query heads, 8 KV heads)
- Llama 3.2 rotary position embeddings with frequency scaling
- Custom CUDA kernels for embedding lookup, RMSNorm, RoPE, causal masking,
  softmax, SiLU multiplication, residual addition, and paged decode attention
- BF16 projection and attention GEMMs through cuBLAS
- A block-table-based paged KV cache with 16-token blocks
- Request admission, page allocation/reuse, decode stepping, and retirement
- A long-row decode softmax supporting sequences up to 2,048 tokens
- CUDA correctness testing and reproducible throughput/memory benchmarks

## Execution overview

```text
token IDs
   |
embedding lookup
   |
16 x [RMSNorm -> QKV -> RoPE -> GQA -> output projection -> residual
      -> RMSNorm -> SwiGLU MLP -> residual]
   |
final RMSNorm -> tied embedding projection -> greedy argmax
```

Prefill computes attention over the complete prompt and writes each layer's K/V
vectors into the paged cache. Decode processes one new token at a time. Its
attention kernel follows the request's logical-to-physical block table directly,
so K/V data does not need to be gathered into a contiguous buffer first.

## Model

The implementation is specialized for the BF16
`meta-llama/Llama-3.2-1B-Instruct` architecture:

| Parameter | Value |
| --- | ---: |
| Transformer layers | 16 |
| Hidden size | 2,048 |
| Intermediate size | 8,192 |
| Query heads | 32 |
| KV heads | 8 |
| Head dimension | 64 |
| Vocabulary size | 128,256 |
| Maximum prompt length | 512 |
| Maximum sequence length | 2,048 |

Obtain the model through an authorized source after accepting Meta's Llama 3.2
license. Place the unsharded BF16 weights at the repository root with this exact
name:

```text
model.safetensors
```

The weights are intentionally excluded from Git.

## Requirements

- Linux with an NVIDIA GPU
- CUDA Toolkit 12.x (including `nvcc` and cuBLAS)
- CMake 3.24 or newer
- Ninja or another CMake-supported build tool
- A GPU with BF16 support and enough memory for the approximately 2.5 GB model
  plus runtime allocations
- Python 3 and `transformers` only for the optional tokenizer helper

The CUDA target defaults to the native GPU architecture, so configure and build
on the machine where the executable will run.

## Build and run

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/my-tiny-vllm
```

The executable currently reads `model.safetensors` from its working directory.
Prompt token IDs and request count are defined in `src/main.cpp`; generated token
IDs are printed to standard output. To tokenize or decode text while developing:

```bash
python -m pip install transformers
python python/tokenizer.py "The capital of France is"
python python/tokenizer.py --decode --ids 791 6864 315 9822 374
```

## Tests

```bash
cmake -S . -B build -G Ninja -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

The current CUDA test checks decode softmax across short, irregular, and
greater-than-1,024-token rows against a CPU reference.

## Benchmarks

Single-request inference was measured on an RTX 4090 with CUDA 12.8, Release
builds, the same five-token prompt, one 16-token warm-up, and three 256-token
runs. Values are medians.

| Revision | KV/execution path | Startup to first token | Decode throughput |
| --- | --- | ---: | ---: |
| Contiguous baseline | Contiguous KV | 2.591 s | 160.9 tokens/s |
| Paged execution refactor | Paged KV | 2.392 s | 241.5 tokens/s |
| Long-row softmax | Paged KV | 2.549 s | 241.9 tokens/s |

The execution and paged-cache refactor improved decode throughput by 1.50x on
this workload. That revision changed more than the cache layout, so the result
does not isolate paged attention as the cause. All revisions generated identical
token sequences in the measured runs.

Reproduce the timing benchmark with:

```bash
python bench/profile_tokens.py ./build/my-tiny-vllm --tokens 256 --repeats 3
```

See [`bench/results-4090.md`](bench/results-4090.md) for methodology and
[`bench/results-memory-a40.md`](bench/results-memory-a40.md) for KV-cache memory
accounting. The latter also documents why the current 1,024-block pool is
overprovisioned for a single active request.

## Current limitations

- Decode is not a fused GPU batch.
- Prompts are compiled into the executable rather than accepted through a CLI or
  service interface.
- Decoding uses greedy argmax, and logits are copied to the CPU for selection.
- The implementation is fixed to one Llama architecture and one BF16 tensor
  layout; it does not parse model configuration dynamically.
- The KV pool is statically sized and currently reserves more memory than the
  single-request configuration requires.
- Kernel launches and CUDA/cuBLAS calls do not yet have comprehensive error
  checking.
- The engine has not been optimized to production vLLM performance or validated
  across GPUs and long-running workloads.

## Repository layout

```text
src/main.cpp                 model loading, transformer execution, scheduler
src/kernels.cu               CUDA kernels and launch wrappers
src/kernels.cuh              kernel wrapper declarations
tests/decode_softmax.cu      CUDA softmax correctness test
python/tokenizer.py          Hugging Face tokenizer helper
bench/profile_tokens.py      token latency and throughput harness
bench/profile_memory.py      process GPU-memory sampler
bench/cache_memory.cu        KV allocation and block-fragmentation probe
```

## Acknowledgements

Built while following [Jed Maczan's tiny-vllm](https://github.com/jmaczan/tiny-vllm)
guide as a learning reference. This implementation follows many of the same
concepts and milestones, while diverging in parts of the execution path, paged
KV-cache implementation, scheduling, testing, and benchmarking.
