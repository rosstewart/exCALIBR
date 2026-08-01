← [Back to README](../README.md)

# GPU Acceleration

ExCALIBR supports JAX-batched GPU fitting via `--device gpu` or `--device cuda:N`.

## What it does

The bootstrap fitting step (Step 1) is the dominant cost. On CPU, bootstrap fits are distributed across cores with joblib. On GPU, all fits for a given bootstrap round are assembled into a single batched tensor and dispatched to JAX as one vectorized call — no process spawning, no inter-process communication.

## What is being GPU-accelerated, and why it is not simple to parallelize

The fitting kernel is a skew-normal mixture EM algorithm. Unlike typical deep-learning workloads (matrix multiplications on fixed-shape tensors), the EM algorithm is:

- **Iterative with convergence checks** — each fit runs for as many steps as needed until the parameter estimates stabilize; fits in the same batch can converge at different steps
- **Stateful across iterations** — the E-step computes posterior responsibilities, the M-step updates parameters, and both feed back into the next iteration; there is no obvious way to pipeline iterations
- **Constraint-enforcing** — mixture weights must stay normalized, component locations must maintain ordering, and the EM must detect and handle label-switching across iterations

JAX JIT-compiles the full batched EM loop (including the convergence check, which becomes a `jax.lax.while_loop`) into CUDA kernels. This gives real speedup, but the iterative convergence structure means the GPU is doing fundamentally different work than a feedforward neural network — the benefit is parallelism across fits in the batch, not just raw floating-point throughput.

## JIT compilation overhead

JAX traces and compiles the EM kernel on the **first call** in a Python session. This compilation is included in the wall-clock time for `run_pipeline.py` (which runs one dataset per process) but is amortized across datasets in the HPC batch runner (`hpc/run_local_array_gpu.sh`), which deliberately keeps one Python process alive across all array tasks so the compiled kernel is reused.

## Benchmarks

Timed on `example/MSH2_Jia_2021.csv` (1,579 variants), `--preset light` (20 bootstraps × 8 fits = 160 fit units), 3-component model. Device: **Tesla V100S-PCIE-32GB**.

| Pass | Step 1 (bootstrap fitting) | Rest of pipeline | Total | Fits/second |
|---|---|---|---|---|
| jit+run (first dataset in session) | 307.6s | 26.8s | 334.4s | 0.52 |
| steady-state (subsequent datasets, same process) | 274.8s | 25.0s | 299.8s | 0.58 |
| **JIT compilation overhead** | **32.8s** | — | — | — |

For comparison, the CPU speed estimates in [Configuration Options](configuration.md#speed-estimates) show `--preset light` on MSH2 taking approximately 10 min at 64 cores. The V100S reduces this to ~5.6 min (jit+run), or ~5 min (steady-state in HPC batch context).

### Hardware context

The V100S-PCIE-32GB is a 2019-era data center GPU. The EM algorithm is memory-bandwidth-sensitive (iterative reads/writes to per-fit parameter tensors) more than raw FLOP-limited. Expected relative performance on more recent hardware:

| GPU | Generation | FP32 TFLOPS | HBM bandwidth | Expected Step 1 speedup vs V100S |
|---|---|---|---|---|
| Tesla V100S-PCIE-32GB (tested) | 2019 | 16.4 | 1,134 GB/s | 1× (baseline) |
| A100-SXM-80GB | 2020 | 19.5 | 2,039 GB/s | ~1.5–2× |
| H100-SXM-80GB | 2022 | 51.2 | 3,350 GB/s | ~2–4× |
| RTX 4090 | 2022 | 82.6 | 1,008 GB/s | ~1× (lower bandwidth than A100/H100) |

Speedup is approximate; the actual gain depends on how memory-bound vs. compute-bound the EM batch is at a given batch size. The RTX 4090 has very high FP32 throughput but lower memory bandwidth than data-center parts, which likely limits its advantage for this workload.

## Usage

```bash
# GPU (default device)
python run_pipeline.py --dataset example/MSH2_Jia_2021.csv --name MSH2_Jia_2021 --device gpu

# Pin to GPU 1 (e.g. GPU 0 is occupied by another job)
python run_pipeline.py --dataset example/MSH2_Jia_2021.csv --name MSH2_Jia_2021 --device cuda:1
```

`run_pipeline.py` runs one dataset per invocation, so only one GPU is ever used. For calibrating many datasets with GPU acceleration, use the HPC batch workflow (`hpc/run_local_array_gpu.sh`), which keeps one Python process alive across all array tasks so the JIT-compiled kernel is reused for every dataset after the first.

## Environment setup

Use the GPU conda environment instead of the standard one:

```bash
conda env create -f excalibr-gpu.yml
conda activate excalibr
pip install -e .
```

The GPU environment installs `jax[cuda12]==0.4.38`, which bundles the required CUDA runtime libraries. No `module load cuda` is needed. See `excalibr-gpu.yml` for details.
