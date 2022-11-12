import numpy as np
import torch
import torchvision
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear,Conv2d,ReLU,MaxPool2d
import matplotlib.pyplot as plt

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#hyper parameters
num_epochs=5
batch_size=4
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
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

# def imshow(img):
#     # img=img/2+0.5 #unnormalize
#     img_np=img.numpy()
#     print(img_np.shape)
#     plt.imshow(np.transpose(img_np,(1,2,0)))
#     plt.show()
    
    
# # get some random trainning images
# dataiter=iter(train_loader)
# images,labels = dataiter.next()
# imshow(torchvision.utils.make_grid(images))
## creating CNN
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1=Conv2d(1,14,5)
        self.pool=MaxPool2d(2,2)
        self.conv2=Conv2d(14,28,5)
        self.fc1=Linear(1792,150)
        self.fc2=Linear(150,100)
        self.fc3=Linear(100,10)
    
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,1792)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
#create varibale for CNN class
model=MNIST_CNN().to(device)
#loss and optimizer functions
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
n_total_steps=len(train_loader)
# training loop
for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        images=images.to(device)
        labels=labels.to(device)
        
        #forward pass
        outputs=model(images)
        print(outputs.shape,labels.shape)
        # loss=1#criterion(outputs,labels)
        
        # #backward and optimizer
        # optimizer.zero_grad()
        # # loss.backward()
        # optimizer.step()
        # if (i+1)%5000==0:
        #   print(f"Epoch [{epoch+1}/{num_epochs}], Step[{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")  

    