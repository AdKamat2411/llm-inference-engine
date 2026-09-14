# RTX 4090 single-request benchmark

Measured on 2026-09-14 with one NVIDIA GeForce RTX 4090, CUDA 12.8.93,
Release builds (`-O2`), and the same Llama 3.2 1B Instruct weights and
five-token prompt. Each binary had one 16-token warm-up run, followed by three
256-token runs. Values below are medians of the three measured runs.

| Revision | KV layout | Startup to first token | Decode throughput |
| --- | --- | ---: | ---: |
| `c7c72ce` | Contiguous | 2.591 s | 160.9 tokens/s |
| `03ea8e7` | Paged | 2.392 s | 241.5 tokens/s |
| `96791bb` | Paged, long-row softmax | 2.549 s | 241.9 tokens/s |

The paged/refactor revision was 1.50x as fast as the contiguous revision on
this workload. This commit changes more than the KV layout, so the result does
not isolate the effect of paging alone. The long-row softmax revision is a
correctness/capacity change; at 256 tokens its measured throughput was within
run-to-run noise of `03ea8e7`.

All three revisions generated the same 256 token IDs in every measured run
(SHA-256 of comma-separated IDs:
`35db841190ed0e8356b810229e7977dc1e7b6d14f81a3ae57cbc0c2a3a4da617`).
The current uncommitted two-request scheduler build also completed 200 tokens
and exited successfully; its first 100 IDs matched the contiguous baseline.

`bench/profile_tokens.py` timestamps line-buffered token output. Decode
throughput is 255 tokens divided by the elapsed time between the first and
256th token lines. It includes CPU argmax and output handling, not just GPU
kernels. Startup-to-first-token includes process startup, safetensors loading,
CUDA initialization, and prefill; it is not an isolated prefill latency.
These are batch-size-one numbers, not a vLLM comparison or a batched-decode
benchmark.

To repeat on a CUDA machine, build each revision in Release mode with the same
model file in its working directory, then run:

```bash
python3 bench/profile_tokens.py /absolute/path/to/build/my-tiny-vllm --tokens 256 --repeats 3
```
