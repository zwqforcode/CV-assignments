import torch
from torch import nn
from torch import ByteStorage
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import TensorDataset,DataLoader, Dataset
from torchvision.datasets import ImageFolder
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():   

    myTransforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    batch_size = 64
    epoch = 32
    
     #  load
    train_dataset = torchvision.datasets.CIFAR10(root='./cifar-10-python/', train=True, download=False, transform=myTransforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    test_dataset = torchvision.datasets.CIFAR10(root='./cifar-10-python/', train=False, download=False, transform=myTransforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)


    # freeze
    alexnet = models.alexnet(pretrained=True)
    for param in alexnet.parameters():
        param.requires_grad = False
 
    # fix FCN into 10
    inchannel = 256 * 6 * 6
    alexnet.classifier = nn.Sequential(
            nn.Linear(inchannel,inchannel//2),  
            nn.ReLU(),
            nn.Linear(inchannel//2,10),  
        )
    
    print(alexnet)
    # check parameters
    total_params = sum(p.numel() for p in alexnet.parameters())
    print('total parameters:{}'.format(total_params))
    total_trainable_params = sum(p.numel() for p in alexnet.parameters() if p.requires_grad)
    print('parameters need to be trained:{}'.format(total_trainable_params))
 
    alexnet = alexnet.to(device)
    criterion = nn.CrossEntropyLoss()
    # only optimize changed parameters
    optimizer = torch.optim.Adam(alexnet.classifier.parameters(), lr=1e-3, weight_decay=1e-3)  # 优化器
    # train
    train(alexnet, train_loader, test_loader, epoch, optimizer, criterion)
 
 
# compute acc
def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total
 
 
# train
def train(model, train_data, test_data, num_epochs, optimizer, criterion):
    best = 0
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        model = model.train()
        
        for im, label in train_data:
            im = im.to(device) 
            label = label.to(device) 
            # forward
            optimizer.zero_grad()
            output = model(im)
            loss = criterion(output, label)
            # backward
            
            loss.backward()
            optimizer.step()
 
            train_loss += loss.item()
            train_acc += get_acc(output, label)
 
        if test_data is not None:
            with torch.no_grad():
                test_loss = 0
                test_acc = 0
                model = model.eval()
                
                for im, label in test_data:
                    im = im.to(device)  
                    label = label.to(device)  
                    output = model(im)
                    loss = criterion(output, label)
                    test_loss += loss.item()
                    test_acc += get_acc(output, label)
                    
                acc = test_acc / len(test_data)   
                if acc > best:
                    torch.save(model, 'trans_alex.pth')
                    best = acc
                    print("best model at epoch {}-----test accuracy is:{}" .format(epoch,acc))
                epoch_str = (
                        "Epoch %d. Train Loss: %f, Train Acc: %f, Test Loss: %f, Test Acc: %f, "
                        % (epoch, train_loss / len(train_data),
                        train_acc / len(train_data), test_loss / len(test_data),
                        test_acc / len(test_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
  
        print(epoch_str )
 
 
if __name__ == '__main__':
    main()