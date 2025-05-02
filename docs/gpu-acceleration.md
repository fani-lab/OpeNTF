# GPU Acceleration in OpeNTF-main

This document outlines the implementation and configuration of GPU acceleration for the computationally intensive `teamsvecs` sparse matrix generation process in `src/cmn/team.py`.

## Overview

The generation of team-skill, team-member, and team-location sparse matrices (`teamsvecs`) can be time-consuming for large datasets. This implementation leverages PyTorch and CUDA-enabled GPUs to accelerate this process.

Key Features:

- **Device Selection:** Supports CPU, single GPU, or multi-GPU execution based on configuration.
- **Batched Processing:** Processes teams in batches on the GPU to manage VRAM usage, crucial for GPUs with varying memory capacities (e.g., 16GB to 80GB).
- **Optimized Data Transfer:** Uses pinned CPU memory for faster transfers to the GPU.
- **Multi-GPU Parallelism:** Distributes the workload across multiple specified GPUs using `torch.multiprocessing`.
- **Error Handling:** Includes specific checks and logging for CUDA Out-of-Memory (OOM) errors.

## Implementation Details (`src/cmn/team.py`)

- **`get_devices(config_str)`:** Parses the acceleration configuration string and returns a list of `torch.device` objects to use.
- **`bucketing_gpu_batched(...)`:** The core GPU processing function. It handles batch size estimation, loops through data in batches, creates sparse tensors on the target GPU using PyTorch, and includes memory management (pinned memory, cache clearing) and OOM error handling.
- **`_bucketing_worker(...)`:** A helper function for multi-GPU processing. Each worker runs `bucketing_gpu_batched` on a subset of the data on its assigned GPU.
- **`gen_teamsvecs(...)`:** The main orchestrator method.
  - Parses configuration (`acceleration`, `gpu_batch_size_mb`).
  - Calls `get_devices`.
  - Branches execution based on devices found (CPU, single GPU, multi-GPU).
  - For multi-GPU, it splits data, launches workers, aggregates results (summing partial sparse tensors), and coalesces the final tensor.
  - Converts the final PyTorch sparse tensor (from GPU paths) back to a SciPy sparse matrix for saving.

## Configuration

The GPU acceleration behavior is controlled via parameters passed in the `cfg` object to `Team.gen_teamsvecs`.

- **`cfg.acceleration` (String):** Specifies the desired processing device(s).
  - `"cpu"`: Use CPU only (serial processing unless multiprocessing is triggered by `cpu:N` format, handled by original logic).
  - `"gpu"`: Use all available CUDA GPUs (if any), otherwise fallback to CPU.
  - `"gpu:0"`: Use only the GPU with ID 0.
  - `"gpu:0,1,3"`: Use GPUs with IDs 0, 1, and 3.
  - _Default (if not specified or invalid):_ `"gpu"`
- **`cfg.gpu_batch_size_mb` (Integer):** Target VRAM usage per batch in Megabytes (MB) for the `bucketing_gpu_batched` function. The actual number of teams per batch is estimated based on this target.
  - _Default (if not specified or invalid):_ `1024` (i.e., 1 GB)

**Example Configuration Snippet (Conceptual):**

```python
# In the script calling Team.gen_teamsvecs
config = SimpleNamespace(
    # ... other config ...
    acceleration="gpu:0,1",  # Use first two GPUs
    gpu_batch_size_mb=2048, # Target 2GB VRAM per batch
    location="country",
    bucket_size=1000 # Still used by CPU path? Review necessity
)

Team.gen_teamsvecs(datapath="...", output="...", cfg=config)
```

## Usage Notes

- **Dependencies:** Requires PyTorch (`torch`) installed with CUDA support matching your GPU drivers.
- **VRAM:** Monitor GPU VRAM usage. If OOM errors occur, decrease `gpu_batch_size_mb` in the configuration.
- **Multi-GPU:** The multi-GPU implementation splits teams across GPUs and sums the resulting sparse tensors. Ensure this aggregation logic (summing) is appropriate for the desired outcome.
- **CPU Fallback:** If CUDA is unavailable or no valid GPUs are specified, the process automatically falls back to the CPU (either serial or the original multiprocessing implementation if `cpu:N` was specified).
