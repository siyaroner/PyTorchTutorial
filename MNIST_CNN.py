import numpy as np
import torch
import torchvision
from torchvision import datasets,transforms
import torch.nn as nn
from torch.nn import Linear,Conv2d,ReLU,MaxPool2d
from torch.optim import Optimizer
import matplotlib.pyplot as plt

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#hyper parameters
num_epochs=5
batch_size=5
learning_rate=0.001
#transform
transform=transforms.ToTensor()

# downloading MNIST dataset
train_dataset=datasets.MNIST(root="./data",
                         train=True,
                         download=True, 
                         transform=transform)
test_dataset=datasets.MNIST(root="./data",
                         train=False,
                         download=True, 
                         transform=transform)
# loading MNIST dataset
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=False)

classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

def imshow(img):
    # img=img/2+0.5 #unnormalize
    img_np=img.numpy()
    print(img_np.shape)
    plt.imshow(np.transpose(img_np,(1,2,0)))
    plt.show()
    
    
# get some random trainning images
dataiter=iter(train_loader)
images,labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1=Conv2d(14,14,5)
        self.pool=MaxPool2d(2,2)
        self.conv2=Conv2d(14,28,5)
        self.fc1=Linear(28*5*5,150)
        self.fc2=Linear(150,100)
        self.fc3=Linear(100,10)
    
    def forwar(self,x):
        x=self.pool(self.ReLU(self.conv1(x)))
        x=self.pool(self.ReLU(self.conv2(x)))
        x=x.view