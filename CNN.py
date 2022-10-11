import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
batch_size=4
trainset=torchvision.datasets.CIFAR10(root="./data/CIFAR10", train=True,download=True,transform=transform)
trainloader= torch.utils.data.DataLoader(trainset,shuffle=True,batch_size=batch_size,num_workers=4)

testset=torchvision.datasets.CIFAR10(root="./data/CIFAR10", train=False,download=True,transform=transform)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
if __name__ == '__main__':
    dataiter=iter(trainloader)
    images,labels=dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,3)
        self.conv2 = nn.Conv2d(6,9,3)
        self.conv3 = nn.Conv2d(9,12,3)
        self.pool=nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(12,12,3)
        self.fc1= nn.Conv2d(12*3*3,10)
    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        x=F.relu(self.conv3(F.relu(self.conv2(x))))
        x=F.relu(self.conv4(self.pool(x)))
        x=x.view(-1,12*3*3) #Flatten
        x=self.fc1(x)
        x=F.softmax(x)
        return x