import numpy as np
import torch
import torchvision
import pickle
from PIL.Image import Image
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import TensorDataset,DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch.nn.functional as F
import os

 
class mymodel(nn.Module):
    def __init__(self):
        super(mymodel,self).__init__()

        self.conv_unit=nn.Sequential(
            nn.Conv2d(3,6,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            nn.Conv2d(6,16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),)
     
        self.fc_unit=nn.Sequential(
            nn.Linear(16*16*16,8*16*16),  
            nn.ReLU(),
            nn.Linear(8*16*16,4*16*16),  
            nn.ReLU(),
            nn.Linear(4*16*16,16*16),
            nn.Linear(16*16,10),
        )
     
    def forward(self,x):
        batchza=x.size(0) 
 
        x=self.conv_unit(x)
        
        x=x.view(batchza,16*16*16)
      
        logits=self.fc_unit(x)

        return logits 
 
 
def train(epoch,train_loader):
    model.train()
    for batch_idx, (data,target) in enumerate(train_loader):
        data,target = data.to(device),target.to(device)
        optimizer.zero_grad()
        criterion=nn.CrossEntropyLoss().to(device) 
        output=model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred=output.argmax(dim=1)
        

 
def test(test_loader):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for data,target in test_loader:
            data,target = data.to(device),target.to(device)
 
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()
            pred = output.argmax(dim=1)
 
            correct += torch.eq(pred,target).float().sum().item()

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
        acc = 100. * correct / len(test_loader.dataset)
        return acc
 
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

if __name__ == '__main__':
    
    model = mymodel().to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    print(model)
    myTransforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    batch_size = 64
    epochs = 20
    
     #  load
    train_dataset = torchvision.datasets.CIFAR10(root='./cifar-10-python/', train=True, download=False, transform=myTransforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.CIFAR10(root='./cifar-10-python/', train=False, download=False, transform=myTransforms)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)


    best = 0
    best_epo = 0
    for epoch in range(1, epochs + 1):  
        train(epoch,train_loader)
        acc = test(test_loader)
        if acc > best:
            torch.save(model, 'mymodel.pth')
            best = acc
            best_epo = epoch
            print("best model at epoch {}-----test accuracy is:{}" .format(epoch,acc))
    
    print("finish, best epoch at {}------test accuracy is: {}" .format(best_epo,best))