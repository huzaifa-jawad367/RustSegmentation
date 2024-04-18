import torch

if torch.cuda.is_available():
    print("GPU is available!")
else:
    print("GPU is not available.")

print(torch.__version__)

print(torch.cuda.memory_summary())