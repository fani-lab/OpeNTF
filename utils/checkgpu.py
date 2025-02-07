import torch

# Check if CUDA is available
print(torch.cuda.is_available())

# If CUDA is available, check the current device
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))  # prints the name of the GPU
    print(torch.cuda.current_device())    # prints the index of the current device
