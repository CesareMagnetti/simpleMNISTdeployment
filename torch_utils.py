import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import io
import numpy as np
from torchvision.transforms.transforms import Grayscale


class View(nn.Module):
    """
    Simple module to `View` a tensor in a certain size

    Wrapper of the `torch.Tensor.View()` function into a forward module
    Args:
        size (tuple) output tensor size
    """

    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

# Network class
class Net(nn.Module):
    def __init__(self, input_size = (28,28), in_channels = 1, out_classes = 10, 
                 kernel_size = 4, stride = 2, padding = 1, dilation = 1,
                 conv_layers = 2, linear_layers = 2, expansion = 5):
        """
        Simple class for a variable sized CNN, defaults to 4 layer MNIST classifier.

        Args:
            input_size (tuple, shape HxW): dimension of input image. default = (28,28), MNIST images.
            in_channels (int): number of channels of input images. default = 1, grayscale MNIST images.
            out_classes (int): number of classes in the classification problem. default = 10, MNIST classes.
            kernel_size (int): kernel size of the convolutional layers. default = 4.
            stride (int): stride of the convolutional layers. default = 2.
            padding (int): padding of the convolutional layers. default = 1.
            dilation (int): dilation of the convolutional layers. default = 1.
            conv_layers (int): number of convolutional layers. default = 2.
            linear_layers (int): number of linear layers. default = 2.
            expansion (int): expansion/reduction factor for channels in conv layers
                             and nodes in linear layers. default = 5 -> expands/reduces as powers of 5.
                             NOTE: for any CNN with more than 4 conv layers this expansion factor
                                   will be impractical. (1->5->25->125->625->3125), more suited expansion
                                   for large CNN is 2: i.e. for RGB images: 3 -> 3*2 -> 3*4 -> 3*8 ... -> 3*2^(n-1)
        """

        super(Net, self).__init__()
        
        # ===== Convolutional Layers ======
        conv = []
        H,W = input_size
        sizes = [input_size,]
        for c in range(conv_layers):
            H = (H+2*padding-dilation*(kernel_size-1)-1)/stride+1
            W = (W+2*padding-dilation*(kernel_size-1)-1)/stride+1
            sizes.append((H,W))
            conv.append(nn.Conv2d(in_channels*(expansion**c), expansion**(c+1), kernel_size, stride, padding))
            conv.append(nn.BatchNorm2d(expansion**(c+1)))
            if c<(conv_layers-1): #don't append ReLU or dropout on last conv layer
                conv.append(nn.ReLU(inplace=True))
                conv.append(nn.Dropout2d(p=0.1))

        self.conv = nn.Sequential(*conv)

        # ===== Reshape from BxCxHxW to BxN =====
        N = int(np.prod(sizes[-1]+(expansion**(c+1),)))
        self.view = View((-1,N))

        # ===== Linear Layers =====
        linear = []
        for l in range(linear_layers):
            if int(N/expansion)>out_classes:
                linear.append(nn.Linear(N,int(N/expansion)))
                linear.append(nn.ReLU(inplace=True))
                linear.append(nn.Dropout(p=0.5))
                N = int(N/expansion)
        linear.append(nn.Linear(N, out_classes))
        linear.append(nn.Sigmoid())

        self.linear = nn.Sequential(*linear)

    
    def forward(self, input_tensors):
        y = self.conv(input_tensors)
        y = self.view(y)
        y = self.linear(y)
        return y


# define device (CPU versus GPU)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


# ====== FUNCTIONS TO BE USED BY app.py =======

# transfor uploaded image to tensor and preprocess it
class transformImage(object):
    def __init__(self, image_size = (28,28), normalisation = (0.1307, 0.3081)):
        self.transform = transforms.Compose([transforms.Grayscale(),
                                             transforms.ToTensor(),
                                             transforms.Resize(size=image_size), # default is PIL.Image.BILINEAR interpolation
                                             transforms.Normalize(*normalisation), # mean and std of training set
                                             ])

    def __call__(self, image_bytes):
        # 1 open image bytes
        image = Image.open(io.BytesIO(image_bytes))
        # 2 transform image
        tensor_image = self.transform(image)
        return tensor_image.unsqueeze(0)


class getPrediction(object):
    def __init__(self):
        checkpoint = torch.load("./models/MNIST.tar", map_location="cpu")
        model = Net()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        self.model = model

    def __call__(self, tensor_image):
        output = self.model(tensor_image)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()
