import torch

if torch.cuda.is_available():
    print("GPU is available!")
else:
    print("GPU is not available.")

print(torch.__version__)

# import torch
# from pynvml import *

# def print_gpu_utilization():
#     nvmlInit()
#     handle = nvmlDeviceGetHandleByIndex(0)
#     info = nvmlDeviceGetMemoryInfo(handle)
#     print(f"GPU memory occupied: {info.used//1024**2} MB.")


# def print_summary(result):
#     print(f"Time: {result.metrics['train_runtime']:.2f}")
#     print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
#     print_gpu_utilization()

# torch.ones((1, 1)).to("cuda")
# print_gpu_utilization()