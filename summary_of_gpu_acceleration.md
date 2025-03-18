# GPU Acceleration in Filter Functions

## Summary of Changes

We've added GPU acceleration to several key filter functions in the `src/cmn_v3/filter_functions` directory:

1. **remove_dup_teams.py**: Finding duplicate teams based on member and skill patterns
2. **filter_min_skills_team.py**: Filtering teams with too few skills
3. **filter_max_skills_team.py**: Filtering teams with too many skills
4. **remove_empty_skills_teams.py**: Removing teams with no skills
5. **filter_max_teams_per_member.py**: Limiting members to a maximum number of teams
6. **filter_min_team_size.py**: Filtering teams with too few members
7. **filter_max_team_size.py**: Filtering teams with too many members
8. **remove_empty_experts_teams.py**: Removing teams with no members
9. **filter_min_teams_per_expert.py**: Filtering members in too few teams

## Implementation Overview

Each filter now follows a common pattern:

1. **Auto-detection**: Automatically detects if a GPU is available
2. **Adaptive batch sizing**: Sets batch sizes based on available GPU memory
3. **Batched processing**: Processes data in batches to avoid memory issues
4. **Memory management**: Clears GPU cache between operations
5. **Fallback mechanism**: Falls back to CPU if GPU memory is exceeded
6. **CPU alternative**: Maintains the original CPU implementation as fallback

## Technical Improvements

### 1. Memory Optimization

- **Chunked Processing**: Large datasets are processed in batches to avoid GPU memory limits
- **Adaptive Sizing**: Batch sizes adjust based on GPU memory (10K-100K for high-end GPUs)
- **Sparse Processing**: Special handling for sparse matrices
- **Memory Cleanup**: Explicit cache clearing between operations

### 2. Performance Optimizations

- **Tensor Operations**: Uses PyTorch tensor operations for faster calculations
- **Parallel Execution**: Takes advantage of GPU's parallel execution model
- **Targeted Acceleration**: GPU is used for expensive operations (matrix sums, non-zero searches)
- **Hybrid Approach**: Some parts use CPU for sparse matrix manipulation after GPU identifies candidates

### 3. Robustness Features

- **OOM Protection**: Gracefully recovers from out-of-memory errors
- **Mixed-Precision**: Handles both dense and sparse matrices
- **Type Flexibility**: Handles various matrix formats (CSR, LIL, dense arrays)
- **Progress Feedback**: Enhanced progress reporting

## Libraries Used

### GPU-specific Libraries

- **PyTorch (`torch`)**: Core tensor operations and GPU management
  - `torch.cuda`: GPU detection and memory management
  - `torch.tensor`: Converting numpy arrays to GPU tensors
  - `torch.sum`: Efficient parallel reduction operations

### Common Libraries

- **NumPy (`numpy`)**: Array operations for CPU implementation and result handling
- **SciPy (`scipy.sparse`)**: Sparse matrix operations
- **tqdm**: Progress tracking for both CPU and GPU implementations
- **time**: Performance measurement
- **gc**: Garbage collection for memory management

### Key Differences in Implementation

- **GPU**: Uses PyTorch's CUDA tensors for matrix operations and leverages batch processing
- **CPU**: Uses NumPy and SciPy's sparse matrix operations directly

## Expected Performance Improvements

| Filter Type          | Dataset Size           | CPU Time | GPU Time | Speedup |
| -------------------- | ---------------------- | -------- | -------- | ------- |
| Duplicates           | 1M teams               | Hours    | Minutes  | 10-50x  |
| Min/Max Skills       | 1M teams               | Minutes  | Seconds  | 5-20x   |
| Empty Teams          | 1M teams               | Minutes  | Seconds  | 5-15x   |
| Teams Per Member     | 1M teams, 100K members | Hours    | Minutes  | 10-30x  |
| Min/Max Team Size    | 1M teams               | Minutes  | Seconds  | 5-15x   |
| Empty Experts        | 1M teams               | Minutes  | Seconds  | 5-15x   |
| Min Teams Per Expert | 1M teams, 100K members | Hours    | Minutes  | 10-30x  |

## GPU Memory Requirements

- **Entry-level GPU (4-8 GB)**: Can process 10K-20K teams per batch
- **Mid-range GPU (10-16 GB)**: Can process 50K teams per batch
- **High-end GPU (24-80 GB)**: Can process 100K+ teams per batch

## Benefits of CPU Mode

The CPU implementation remains important for:

1. **Environments without GPUs**: Cloud instances, CI/CD pipelines, etc.
2. **Very small datasets**: For datasets under ~10K teams, CPU may be faster
3. **Development work**: Easier debugging without GPU memory complications
4. **Extreme sparsity**: When matrices are >99.9% zeros, CPU sparse implementations may be more efficient

## Recommendations for Use

- For datasets >100K teams: GPU acceleration is strongly recommended
- For duplicate detection: GPU acceleration provides the most significant benefit
- For production environments: Configure for the largest batch size that memory allows
- For mixed CPU/GPU environments: The auto-detection ensures optimal use of available resources
