# 7.2 Classification of MNIST Based on RNN

# 1. Code
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.utils.data as Data
from torchvision import transforms
import hiddenlayer as hl
from tqdm import tqdm

def visualRNN(MyRNNimc):
    hl_graph = hl.build_graph(MyRNNimc, torch.zeros([1, 28, 28]))
    hl.graph.theme = hl.graph.THEMES['blue'].copy()
    # print(hl_graph)
    hl_graph.save(path='./Images/7_5_Structure-of-RNN-classifier',format='png')


def visualTraining(train_loss_all, test_loss_all, train_acc_all, test_acc_all):
    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.plot(train_loss_all,'ro-',label='Train Loss')
    plt.plot(test_loss_all,'bs-',label='Val Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(1,2,2)
    plt.plot(train_acc_all,'ro-',label='Train Acc')
    plt.title('Figure 7-6-b Training process of RNN image classifier')
    plt.plot(test_acc_all,'bs-',label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend()
    plt.title('Figure 7-6-b Validation process of RNN image classifier')
    plt.show()
    plt.savefig('./Images/7_6_Training-of-RNN-image-classifier.png')


# data preparation
train_data = torchvision.datasets.MNIST(
    root='./data/Image', train=True, transform=transforms.ToTensor(),
    download=True
)

train_loader = Data.DataLoader(
    dataset=train_data, batch_size=64, shuffle=True,
    # num_workers=2
)

test_data = torchvision.datasets.MNIST(
    root='./data/Image', train=False, transform=transforms.ToTensor(),
    download=True
)
test_loader = Data.DataLoader(
    dataset=test_data, batch_size=64, shuffle=True,
    # num_workers=2
)


# model definition
class RNNimc(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNimc, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim,
                          batch_first=True, nonlinearity='relu')
        self.fc1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, h_n = self.rnn(x, None)
        out = self.fc1(out[:, -1, :])
        return out


# model initialization
input_dim = 28  # number of pixels every line in each image
hidden_dim = 128  # number of neurons of RNN
layer_dim = 1  # number of layers of RNN
output_dim = 10  # dimension of hidden layer (images in 10 classes)

MyRNNimc = RNNimc(input_dim, hidden_dim, layer_dim, output_dim)
print(MyRNNimc)

# visualization of RNN classifier
visualRNN(MyRNNimc)


# training and prediction
optimizer = torch.optim.RMSprop(MyRNNimc.parameters(),lr=0.0003)
criterion = nn.CrossEntropyLoss()
train_loss_all = []
train_acc_all = []
test_loss_all = []
test_acc_all = []
num_epochs = 10
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch,num_epochs-1))

    # set model to train
    MyRNNimc.train()
    corrects = 0
    loss = 0
    train_num = 0
    for step, (b_x,b_y) in tqdm(enumerate(train_loader),ascii='Training: '):
        xdata = b_x.view(-1,28,28)
        output = MyRNNimc(xdata)
        pre_lab = torch.argmax(output,1)
        loss = criterion(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss += loss.item() * b_x.size(0)
        corrects += torch.sum(b_y.data == pre_lab)
        train_num += b_x.size(0)
    # compute loss and accuracy on train dataset after 1 epoch of training
    train_loss_all.append(loss/train_num)
    train_acc_all.append(float(corrects)/train_num)
    print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))

    # set model to evaluate
    MyRNNimc.eval()
    corrects = 0
    loss = 0
    test_num = 0
    for step, (b_x, b_y) in tqdm(enumerate(test_loader),ascii='Testing: '):
        xdata = b_x.view(-1, 28, 28)
        output = MyRNNimc(xdata)
        pre_lab = torch.argmax(output,1)
        loss = criterion(output,b_y)
        loss += loss.item() * b_x.size(0)
        corrects += torch.sum(pre_lab==b_y.data)
        test_num += b_x.size(0)

    # compute loss and accuracy on test set after 1 epoch of training
    test_loss_all.append(loss/test_num)
    test_acc_all.append(float(corrects)/test_num)
    print('{} Test Loss: {:.4f} Test Acc: {:.4f}'.format(
        epoch, test_loss_all[-1], test_acc_all[-1]
    ))

# visualization of the training process
visualTraining(train_loss_all,test_loss_all,train_acc_all,test_acc_all)
```

# 2. Illustration

##  2.1 Data Preparation

```python
# data preparation
train_data = torchvision.datasets.MNIST(
    root='./data/Image', train=True, transform=transforms.ToTensor(),
    download=True
)

train_loader = Data.DataLoader(
    dataset=train_data, batch_size=64, shuffle=True,
    # num_workers=2
)

test_data = torchvision.datasets.MNIST(
    root='./data/Image', train=False, transform=transforms.ToTensor(),
    download=True
)
test_loader = Data.DataLoader(
    dataset=test_data, batch_size=64, shuffle=True,
    # num_workers=2
)
```

Training set: 60000 28 * 28 gray images

Test set: 10000 28 * 28 gray images

## 2.2 Model Definition & Initialization

```python
# model definition
class RNNimc(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNimc, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim,
                          batch_first=True, nonlinearity='relu')
        self.fc1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, h_n = self.rnn(x, None)
        out = self.fc1(out[:, -1, :])
        return out


# model initialization
input_dim = 28  # number of pixels every line in each image
hidden_dim = 128  # number of neurons of RNN
layer_dim = 1  # number of layers of RNN
output_dim = 10  # dimension of hidden layer (images in 10 classes)

MyRNNimc = RNNimc(input_dim, hidden_dim, layer_dim, output_dim)
print(MyRNNimc)


```

```python
def visualRNN(MyRNNimc):
    hl_graph = hl.build_graph(MyRNNimc, torch.zeros([1, 28, 28]))
    hl.graph.theme = hl.graph.THEMES['blue'].copy()
    # print(hl_graph)
    hl_graph.save(path='./Images/7_5_Structure-of-RNN-classifier',format='png')

# visualization of RNN classifier
visualRNN(MyRNNimc)
```


```python
RNNimc(
  (rnn): RNN(28, 128, batch_first=True)
  (fc1): Linear(in_features=128, out_features=10, bias=True)
)
```


![](Images/7_5_Structure-of-RNN-classifier.png)

Figure 7-5 Structure of RNN classifier


## 2.3 Training and Prediction


```python
# training and prediction
optimizer = torch.optim.RMSprop(MyRNNimc.parameters(),lr=0.0003)
criterion = nn.CrossEntropyLoss()
train_loss_all = []
train_acc_all = []
test_loss_all = []
test_acc_all = []
num_epochs = 30
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch,num_epochs-1))

    # set model to train
    MyRNNimc.train()
    corrects = 0
    loss = 0
    train_num = 0
    for step, (b_x,b_y) in tqdm(enumerate(train_loader),ascii='Training: '):
        xdata = b_x.view(-1,28,28)
        output = MyRNNimc(xdata)
        pre_lab = torch.argmax(output,1)
        loss = criterion(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss += loss.item() * b_x.size(0)
        corrects += torch.sum(b_y.data == pre_lab)
        train_num += b_x.size(0)
    # compute loss and accuracy on train dataset after 1 epoch of training
    train_loss_all.append(loss/train_num)
    train_acc_all.append(float(corrects)/train_num)
    print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))

    # set model to evaluate
    MyRNNimc.eval()
    corrects = 0
    loss = 0
    test_num = 0
    for step, (b_x, b_y) in tqdm(enumerate(test_loader),ascii='Testing: '):
        xdata = b_x.view(-1, 28, 28)
        output = MyRNNimc(xdata)
        pre_lab = torch.argmax(output,1)
        loss = criterion(output,b_y)
        loss += loss.item() * b_x.size(0)
        corrects += torch.sum(pre_lab==b_y.data)
        test_num += b_x.size(0)

    # compute loss and accuracy on test set after 1 epoch of training
    test_loss_all.append(loss/test_num)
    test_acc_all.append(float(corrects)/test_num)
    print('{} Test Loss: {:.4f} Test Acc: {:.4f}'.format(
        epoch, test_loss_all[-1], test_acc_all[-1]
    ))


```

In the code above, test set is used to estimate the effect of classification of current network
after each epoch of training.

```python
def visualTraining(train_loss_all, test_loss_all, train_acc_all, test_acc_all):
    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.plot(train_loss_all,'ro-',label='Train Loss')
    plt.plot(test_loss_all,'bs-',label='Val Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(1,2,2)
    plt.plot(train_acc_all,'ro-',label='Train Acc')
    plt.plot(test_acc_all,'bs-',label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend()
    plt.title('Figure 7-6 Training process of RNN image classifier')
    plt.show()
    plt.savefig('./Images/7_6_Training-process-of-RNN-image-classifier.png')


# visualization of the training process
visualTraining(train_loss_all,test_loss_all,train_acc_all,test_acc_all)
```






https://www.361shipin.com/blog/1502002188589735937
