import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms
import os, json

class View(nn.Module):
    """Simple module to `View` a tensor in a certain size

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
    def __init__(self, input_size = (28,28), in_channels = 1, kernel_size = 4,
                 stride = 2, padding = 1, dilation = 1, conv_layers = 2,
                 linear_layers = 2, expansion = 2, out_classes = 10):

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
            if c<(conv_layers-1): #don't append ReLU on last conv layer
                conv.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*conv)

        # ===== Reshape from BxCxHxW to BxN =====
        print(sizes[-1], expansion**(c+1))
        N = int(np.prod(sizes[-1]+(expansion**(c+1),)))
        self.view = View((-1,N))

        # ===== Linear Layers =====
        linear = []
        for l in range(linear_layers):
            if int(N/expansion)>out_classes:
                linear.append(nn.Linear(N,int(N/expansion)))
                linear.append(nn.ReLU(inplace=True))
                N = int(N/expansion)
        linear.append(nn.Linear(N, out_classes))
        linear.append(nn.Sigmoid())

        self.linear = nn.Sequential(*linear)

    
    def forward(self, input_tensors):
        y = self.conv(input_tensors)
        y = self.view(y)
        y = self.linear(y)
        return y


# define some variables for the training routine
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

max_epochs = 10
batch_size_train, batch_size_test = 64, 32
num_workers_train, num_workers_test = 0, 0
lr, momentum = 0.01, 0.5

# define the network
model = Net(expansion=5)
model.to(device)
model.train()
print(model)

# define the optimiser
optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

# define the dataset
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.1307,), std=(0.3081,))]) # mean and std of MNIST training set
train_ds = torchvision.datasets.MNIST('./', train=True, download=True, transform=transform)
test_ds = torchvision.datasets.MNIST('./', train=False, download=True, transform=transform)

# define the data loaders
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size_train, num_workers=num_workers_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size_test, num_workers=num_workers_test, shuffle=False)

# variables to show logs of how the training is going to the console
log_interval_train, log_interval_test = len(train_loader)//10, len(test_loader)//10

# define loss function
criterion = nn.CrossEntropyLoss()

# define variable to store logs
logs = {'train': [], 'test': [], 'accuracy': None}

# train the network
for epoch in range(1, max_epochs+1):

    running_loss= 0
    for batch_idx, data in enumerate(train_loader):

        images, target = data
        images = images.to(device)
        target = target.to(device)

        # forward pass
        output = model(images)
        loss = criterion(output, target)

        # backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()

        # store logs
        running_loss+=loss.item()

        if batch_idx % log_interval_train == 0:
            print("[{}]/[{}] ({:.0f}%)\tloss: {:.6f}".format(epoch, max_epochs, batch_idx/len(train_loader)*100, loss.item()))

    # store average loss
    logs['train'].append(running_loss/len(train_loader))

# test network
model.eval()
running_loss, correct, total = 0, 0, 0
with torch.no_grad():
    for batch_idx, data in enumerate(test_loader):

        images, target = data
        images = images.to(device)
        target = target.to(device)

        # forward pass
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        # store logs
        loss = criterion(output, target)
        running_loss+=loss.item()

        if batch_idx % log_interval_test == 0:
            print("test epoch ({:.0f}%)\tloss: {:.6f}".format(batch_idx/len(test_loader)*100, loss.item()))

accuracy = correct/total
print('accuracy on test set: {:.2f}%'.format(accuracy*100))
# store logs
logs['test'].append(running_loss/len(test_loader))
logs['accuracy'] = accuracy

# save logs
savepath = os.path.join(os.getcwd(), "logs")
if not os.path.exists(savepath):
    os.mkdir(savepath)
with open(savepath + "/logs.json", 'w') as fp:
    json.dump(logs, fp)

# save model
savepath = "models/MNIST.tar"
torch.save({'model_state_dict': model.state_dict()},
           savepath)

