# my-tiny-vllm

Your rewrite of the [tiny-vllm](../tiny-vllm) course. The sibling repo is the answer key — keep `src/` here yours.

## Ground rules

1. Read one section of `../tiny-vllm/README.md`.
2. Implement it here.
3. Only then peek at `../tiny-vllm/src/`.

First milestone: load Llama 3.2 1B Instruct (`model.safetensors`) and emit one token.

## Layout

- `include/json.hpp` — nlohmann/json (vendored, same as the course)
- `python/tokenizer.py` — Hugging Face tokenizer helper (`pip install transformers`)
- `src/main.cpp` / `src/kernels.cu` — empty on purpose

## Build (needs NVIDIA CUDA)

This will not compile on a Mac. When you have a GPU box:

```bash
cmake -B build -G Ninja
cmake --build build
```

You will likely need to point `CMAKE_CUDA_COMPILER` at your `nvcc` in `CMakeLists.txt`.
