# A40 KV-cache memory measurements

Measured on 2026-09-14 with one NVIDIA A40 and CUDA 12.8.93. Both revisions used
the same Llama 3.2 1B Instruct weights, five-token prompt, Release build, and
256 generated tokens. `bench/profile_memory.py` sampled each process's GPU
memory through `nvidia-smi` every 50 ms; three runs per revision returned the
same peak.

| Revision | KV-cache reservation | Peak process GPU memory |
| --- | ---: | ---: |
| `c7c72ce` contiguous | 64 MiB | 2,748 MiB |
| `03ea8e7` paged pool | 512 MiB | 3,196 MiB |

The paged revision reserves 448 MiB more GPU memory in this configuration.
The measured whole-process peak is also 448 MiB higher. A separate
`cudaMemGetInfo` probe measured free-memory deltas of exactly 64 MiB and
512 MiB for the two K/V allocations. These are allocation sizes, not a
claim that paging reduces total VRAM use in this implementation.

The paged pool has 1,024 fixed blocks. Each block holds 16 token positions,
all 16 layers, K and V, and occupies 0.5 MiB. For one request with 260 cached
tokens, 17 blocks (8.5 MiB) are assigned. Live K/V values occupy 8.125 MiB;
12 unused positions in the last block account for 0.375 MiB of internal
fragmentation (4.4% of assigned block space). The other 1,007 blocks
(503.5 MiB) are free for other requests but remain reserved in the CUDA pool.

For request lengths 5, 20, and 512, the same probe reported 35 assigned
blocks (17.5 MiB), 16.781 MiB of live K/V values, and 0.719 MiB of unused
tail space. This is deterministic block accounting, not a workload trace or
an external-fragmentation measurement. With `MAX_ACTIVE_REQUESTS = 1` and
`MAX_SEQ_LEN = 2048`, the configured pool's 16,384 token slots are eight
times the maximum concurrent capacity currently needed.

Nsight Compute 2025.1.1 was present, but even a one-kernel test failed with
`ERR_NVGPUCTRPERM`. The host reported `RmProfilingAdminOnly: 1`; this
container cannot grant itself host GPU-counter access. No `ncu` memory
transaction, cache-hit, or occupancy metrics were collected. NVIDIA documents
the host/container permission requirement at
<https://developer.nvidia.com/ERR_NVGPUCTRPERM>.

Reproduce the allocation and page accounting with:

```bash
nvcc -O2 -o cache_memory bench/cache_memory.cu
./cache_memory 260
./cache_memory 5 20 512
```

Reproduce sampled peak GPU memory with each committed binary and its model
file in the working directory:

```bash
python3 bench/profile_memory.py /absolute/path/to/build/my-tiny-vllm --tokens 256
```
