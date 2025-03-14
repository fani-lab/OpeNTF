import sys
print(f"Python path: {sys.path}")
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else None}")
except ImportError as e:
    print(f"ImportError: {e}")
    print("Detailed error information:")
    import traceback
    traceback.print_exc() 