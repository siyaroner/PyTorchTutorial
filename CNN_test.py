import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#device configuration
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#hyper-parameters
num_epochs=4
batch_size=4
learning_rate=0.001

#dataset has PILImage images of range [0,1].
#We transform them to Tensor of normalized range [-1,1]
transform=transforms.Compose(
    [transforms.ToTensor(),
     transforms.NOrmalize((0.5,0.5,0.5,0.5),(0.5,0.5,0.5,0.5))])

train_dataset=torchvision.datasets.CIFAR10(root=".data/",train=True, download=True,transform=transform)
test_dataset=torchvision.datasets.CIFAR10(root=".data/",train=False, download=True,transform=transform)
train_loader=torchvision.datasets.CIFAR10(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=torchvision.datasets.CIFAR10(test_dataset,batch_size=batch_size,shuffle=False)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3,6,3)
        self.conv2 = nn.Conv2d(6,9,3)
        self.conv3 = nn.Conv2d(9,12,3)
        self.pool=nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(12,12,3)
        self.fc1= nn.Linear(1452,10)
    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        x=F.relu(self.conv3(F.relu(self.conv2(x))))
        x=F.relu(self.conv4(self.pool(x)))
        x=x.view(-1,1452) #Flatten
        x=self.fc1(x)
        x=F.softmax(x)
        return x
    
model=ConvNet()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr= learning_rate)
for epoch in range (num_epochs):
        for i,(images,labels) in enumerate (train_loader):
            #origin shape: [4,3,32,32]= 4,3,1024
            #input_layer=3 input channels, 6 output channels, 5 kernel size
            images=images.to(device)
            labels=labels.to(device)
            #forward pass
            outputs=model(images)
            loss=criterion(outputs,labels)
            #backward and optimize
            optimizer.zero_grad()
            loss.bakcward()
            optimizer.step()
            if i(+1)%2000==0:
                print(f"index= {i+1} epoch= {epoch+1}/{num_epochs} loss={loss.item():.4f}")

print("Traning is finished")