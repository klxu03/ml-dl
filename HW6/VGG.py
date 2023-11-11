## Standard Library
import os
import json

## External Libraries
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as functional
from torch.utils.data import Dataset, DataLoader
from skimage import io
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

## Import VGG and FashionMNIST
from torchvision.models import vgg16
from torchvision.datasets import FashionMNIST

## Specify Batch Size
train_batch_size = 32
test_batch_size = 32

## Specify Image Transforms
img_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

## Download Datasets
train_data = FashionMNIST('./data', transform=img_transform, download=True, train=True)
test_data = FashionMNIST('./data', transform=img_transform, download=True, train=False)

## Initialize Dataloaders
training_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

vgg16_clone = vgg16(weights = None) # randomly initialize
vgg16_clone.classifier[6] = nn.Linear(vgg16_clone.classifier[6].in_features, 10) # change the last layer to 10 classes
vgg16_clone.to(device)

# Define loss func and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(vgg16_clone.parameters(), lr=0.01, momentum=0.9)

# Train the model
vgg16_clone.train()
for epoch in range(5):
    running_loss = 0.0
    for input, label in training_dataloader:
        input = input.cuda()
        label = label.cuda()

        optimizer.zero_grad()
        
        output = vgg16_clone(input)
        loss = criterion(output, label)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        input = input.cpu()
        label = label.cpu()
    
    print(f"EPOCH {epoch+1} - Loss: {running_loss/len(training_dataloader)}")

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    vgg16_clone.eval()
    for input, label in test_dataloader:
        input = input.cuda()
        label = label.cuda()

        output = vgg16_clone(input)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

        input = input.cpu()
        label = label.cpu()

vgg_clone_accuracy = 100 * correct / total
print(f"Accuracy of the VGG16 clone, so pretrained from scratch, model on test images: {vgg_clone_accuracy}%")

# Free model from GPU Ram
vgg16_clone.to(torch.device("cpu"))

vgg16_pre = vgg16(pretrained = True)

# Freeze the parameters in vgg16_pre
for param in vgg16_pre.parameters():
    param.requires_grad = False

vgg16_pre.classifier[6] = nn.Linear(vgg16_pre.classifier[6].in_features, 10)
optimizer = torch.optim.SGD(vgg16_pre.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

vgg16_pre.to(device)

# Train the model
vgg16_pre.train()
for epoch in range(5):
    running_loss = 0
    for input, label in training_dataloader:
        input = input.cuda()
        label = label.cuda()

        optimizer.zero_grad()
        
        output = vgg16_pre(input)
        loss = criterion(output, label)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        input = input.cpu()
        label = label.cpu()
    
    print(f"Epoch {epoch+1} - Loss: {running_loss/len(training_dataloader)}")

# Evaluate the model
vgg16_pre.eval()
correct = 0
total = 0
with torch.no_grad():
    for input, label in test_dataloader:
        input = input.cuda()
        label = label.cuda()

        output = vgg16_pre(input)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

        input = input.cpu()
        label = label.cpu()

vgg_pre_accuracy = 100 * correct / total
print(f"Accuracy of the VGG16 pretrained with initial weights finetuned model on test images: {vgg_pre_accuracy}%")
print(f"Reminder | Accuracy of the VGG16 clone, so pretrained from scratch, model on test images: {vgg_clone_accuracy}%")