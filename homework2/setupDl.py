import torch
import torchvision

print("PyTorch CUDA Version:", torch.version.cuda)
print("Torchvision CUDA Version:", torchvision.__version__)
print("CUDA Available:", torch.cuda.is_available())