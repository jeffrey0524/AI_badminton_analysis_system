import torch

print("GPUs number: ", torch.cuda.device_count(), "GPUs name: ", torch.cuda.get_device_name())