import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io

transform = transforms.Compose([transforms.Normalize(mean=(0.1307,), std=(0.3081,)), # mean and std of MNIST training set
                                ])
