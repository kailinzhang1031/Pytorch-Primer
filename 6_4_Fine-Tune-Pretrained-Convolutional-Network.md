# 6.4 Fine-Tune Pretrained Convolutional Network

## 1. Code 

```python
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import hiddenlayer as hl
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import torch.utils.data as Data
from torchvision import models
from torchvision import transforms
from torchvision.datasets import ImageFolder


def visualImagePerBatch(b_x, b_y, std, mean):
    plt.figure(figsize=(12, 6))
    for ii in np.arange(len(b_y)):
        plt.subplot(4, 8, ii + 1)
        image = b_x[ii, :, :, :].numpy().transpose((1, 2, 0))
        image = std * image + mean
        image = np.clip(image, 0, 1)
        plt.imshow(image)
        plt.title(b_y[ii].data.numpy())
        plt.axis('off')
    plt.subplots_adjust(hspace=0.3)
    plt.show()


path = 'G:/vgg16-397923af.pth'
# import VGG16 network
vgg16 = models.vgg16(pretrained=True)
# get feature extraction layer of VGG16
vgg = vgg16.features
# froze parameters of feature extraction layer, not update
for para in vgg.parameters():
    para.requires_grad_(False)


# construct network by Feature Extraction Layer of VGG16 + Fully Connected Layer
class MyVggModel(nn.Module):
    def __init__(self):
        super(MyVggModel, self).__init__()
        # feature extraction layer of pretrained VGG16
        self.vgg = vgg
        # add new fully connected layer
        self.classifier = nn.Sequential(
            nn.Linear(25088, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.vgg(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output


# output structure of net
Myvggc = MyVggModel()
print(Myvggc)

# train set preprocessing
train_data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# validation set preprocessing
val_data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load images from training set
train_data_dir = 'data/Image/10-Monkey-Species/training'
train_data = ImageFolder(train_data_dir, transform=train_data_transforms)
train_data_loader = Data.DataLoader(train_data,
                                    batch_size=32,
                                    shuffle=True,
                                    # num_workers=2
                                    )

# load images from validation set
val_data_dir = 'data/Image/10-monkey-species/validation'
val_data = ImageFolder(val_data_dir, transform=val_data_transforms)
val_data_loader = Data.DataLoader(val_data, batch_size=32,
                                  shuffle=True,
                                  # num_workers=2
                                  )

print('Instances in training set: ', len(train_data_loader))
print('Instances in validation set: ', len(val_data_loader))

# get data in a batch
for step, (b_x, b_y) in enumerate(train_data_loader):
    if step > 0:
        break

# visualize images in a batch of training set
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

for step, (b_x, b_y) in tqdm(enumerate(train_data_loader)):
    # visualImagePerBatch(b_x,b_y,mean,std)
    # print(b_y)
    if step == 0:
        break

# optimizer definition
optimizer = torch.optim.Adam(Myvggc.parameters(), lr=0.003)
loss_func = nn.CrossEntropyLoss()
historyl = hl.History()
canvasl = hl.Canvas()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print(device)
train_loss_all = []

for epoch in range(50):
    train_loss_epoch = 0
    val_loss_epoch = 0
    train_corrects = 0
    val_corrects = 0
    # iterative computation on training set
    Myvggc.to(device=device)
    Myvggc.train()
    for step, (b_x,b_y) in tqdm(enumerate(train_data_loader),desc=f'Training when epoch at {epoch}'):
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        # compute loss on every batch
        output = Myvggc(b_x)
        loss = loss_func(output, b_y)
        pre_lab = torch.argmax(output,1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item() * b_x.size(0)
        train_corrects += torch.sum(pre_lab==b_y.data)
    # compute loss and accuracy in an epoch
    train_loss = train_loss_epoch / len(train_data.targets)
    train_acc = train_corrects.double() / len(train_data.targets)
    train_loss_all.append(train_loss)
    # prediction on validation set
    Myvggc.eval()
    for step, (val_x, val_y) in tqdm(enumerate(val_data_loader),desc=f'Testing when epoch at {epoch}'):
        val_x = val_x.to(device)
        val_y = val_y.to(device)
        output = Myvggc(val_x)
        loss = loss_func(output,val_y)
        pre_lab = torch.argmax(output,1)
        val_loss_epoch += loss.item() * val_x.size(0)
        val_corrects += torch.sum(pre_lab==val_y.data)

    # compute loss and accuracy in an epoch
    val_loss = val_loss_epoch / len(val_data_loader)
    val_acc = val_corrects.double() / len(val_data_loader)
    # save output loss and accuracy of every epoch
    historyl.log(epoch,train_loss=train_loss,
                 val_loss=val_loss,
                 train_acc=train_acc.item(),
                 val_acc=val_acc.item()
                 )

# visualization of training process
print(historyl['train_loss'])
print(historyl['val_loss'])
with canvasl:
    canvasl.draw_plot([historyl['train_loss'],historyl['val_loss']])
    canvasl.draw_plot([historyl['train_acc'],historyl['val_acc']])
print(train_loss_all)
```


## 2. Illustration


Here we use 10-Monkey-Species dataset from [here](https://www.kaggle.com/datasets/slothkong/10-monkey-species).

This dataset contains 140 * 10(classes) RGB images in training set and 30 * 10(classes) RGB images in test set.

Instances in training set:  35

Instances in validation set:  9

Structure of the network:

```python
MyVggModel(
  (vgg): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=512, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=512, out_features=256, bias=True)
    (4): ReLU()
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=256, out_features=10, bias=True)
    (7): Softmax(dim=1)
  )
)

```