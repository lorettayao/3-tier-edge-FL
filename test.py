import torch
import numpy as np
cuda0 = torch.device('cuda', 0)
from torchvision.datasets import ImageFolder
image_folder = ImageFolder("aaron", transform=None, target_transform=None)
print(image_folder)