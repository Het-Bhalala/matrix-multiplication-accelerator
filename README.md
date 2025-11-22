# matrix-multiplication-accelerator

# GPU & CPU Matrix Multiplication — Observational Report

**Author:** Het Bhalala  
**Field:** GPU & AI (early-career experiments)  
**GPU:** NVIDIA GeForce MX350  
**Languages & Tools:** C, CUDA C/C++, nvcc, Make

---

## What I set out to do (task summary)
- Implement a **CPU matrix multiplication** that is *optimized for memory locality* (tiling / blocking) and supports square sizes from **2×2 up to 8192×8192**, with matrices allocated by `malloc` and filled with random integers.  
- Implement **two CUDA GPU versions**: a naive kernel and a tiled shared‑memory kernel. I verified GPU results against my CPU implementation and explored **tile size / block dim / grid dim** to find an optimal configuration for my hardware. fileciteturn1file0

---

## What I built (files)
- **CPU naive (baseline):** `het.c` — straightforward triple-nested loop, high-precision timing. fileciteturn1file3
- **CPU optimized (tiled):** `optimized_matrix.c` — blocked multiplication (default `BLOCK_SIZE 64` in code). Saves matrices to disk for GPU verification. fileciteturn1file1
- **GPU naive:** `GPU_navie_mul.cu` — direct global-memory multiply (one thread per element).
- **GPU tiled:** `opt_GPU_matrix.cu` — cooperative tiling with shared memory; I tested multiple tile sizes (6, 8, 32, 64) and tuned launch configs. fileciteturn1file0

> Note: Terminal outputs, timing tables, and plots for all runs are compiled in **Assignment-2.pdf** (see charts on *page 5*, and method outputs on *pages 4–5*). fileciteturn1file0

---

## What I observed (my results)

### CPU
- Moving from **naive CPU** to **tiled CPU** dramatically improved locality.  
  **Observation:** For **8192×8192**, the execution time dropped **from ~5401 s to ~613 s** (≈ **9× faster**). fileciteturn1file0

### GPU
- The **naive GPU** suffered from repeated global-memory reads.  
- The **tiled GPU** kernel loaded tiles of A and B into **shared memory**, then computed blocks of P.
- I explored tile sizes **6, 8, 32, 64** and found **T=32** balanced occupancy and reuse on MX350.  
  **Observation:** On large sizes (e.g., 8192×8192), kernel time reduced **from ~32.7 s to ~7.7 s** using the tiled approach. (See the callouts on the two charts on *page 5*.) fileciteturn1file0

**Why T=32 worked best on my GPU:** MX350 supports 48 KB shared memory and 1024 threads/block; with 4‑byte ints, 32×32 tiles for A and B require ~8 KB shared memory per block, keeping within limits while enabling full 32×32=1024 threads/block. fileciteturn1file0

---

## How to build & run

### Prereqs
- CUDA Toolkit (11+), a CUDA-capable GPU, and a C/C++ toolchain.

### Quick compile
```bash
# CPU baselines
gcc het.c -O2 -o cpu_naive
gcc optimized_matrix.c -O3 -o cpu_tiled

# GPU versions
nvcc GPU_navie_mul.cu -O3 -o gpu_naive
nvcc opt_GPU_matrix.cu -O3 -o gpu_tiled
```

### Example run
```bash
# CPU
./cpu_naive
./cpu_tiled

# GPU
./gpu_naive
./gpu_tiled
```

> `optimized_matrix.c` additionally writes `M_<N>.bin`, `N_<N>.bin`, `P_<N>.bin` for verifying GPU results element‑wise. fileciteturn1file1

---

## Notes on verification
I compare the GPU output against the CPU result; if any element mismatches I print **“Verification Failed!”** otherwise **“Verification Success!”** (as required by the task). See screenshots & logs in the PDF. fileciteturn1file0

---

## Extra thought experiment (shared memory headroom)
When shared memory is abundant relative to block size, potential optimizations include:
- **Larger tiles / multi‑tile per block** to increase reuse before committing to global memory.
- **Double buffering** in shared memory to overlap loads with compute.
- **Register tiling** (each thread computes a small output tile, keeping partial sums in registers longer).

These are beneficial because they **reduce global memory traffic**, improve **arithmetic intensity**, and **hide latency** via better pipelining; the exact gains depend on SM occupancy and register pressure on the target GPU.

---

## Plots & raw output
Timing tables and plots (CPU vs GPU naive vs GPU tiled) are shown in my PDF report; see especially the **two charts on page 5** highlighting the 32.7→7.7 s reduction for tiled GPU. fileciteturn1file0
