# 8.3 Image Noise Reduction Based on Convolutional Auto Encoder

# 1. Code


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.util import random_noise
from skimage.measure import _ccomp
import hiddenlayer as hl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import  STL10


def read_image(data_path):
    with open(data_path,'rb') as f:
        data1 = np.fromfile(f,dtype=np.uint8)
        images = np.reshape(data1, (-1,3,96,96))
        # convert the format of images to RGB to visualize
        images = np.transpose(images,(0,3,2,1))
    return images/255.0

def gaussian_noise(images, sigma):
    sigma2 = sigma ** 2/(255*2) # variance
    image_noisy = np.zeros_like(images)
    for ii in range(images.shape[0]):
        image = images[ii]
        # using skimage random_noise to add noise
        noise_im = random_noise(image,mode='gaussian',var=sigma2,clip=True)
        image_noisy[ii] = noise_im
    return image_noisy


def visualImagesWithoutAndWithNoise(images, images_noise):
    plt.figure(figsize=(6,6))
    for ii in np.arange(36):
        plt.subplot(6,6,ii+1)
        plt.imshow(images[ii,...])
        plt.axis('off')
    plt.show()
    plt.title('Figure 8-9-a Images without Noise')
    plt.savefig('./Images/8_9_a_Images_without-noise.png')

    plt.figure(figsize=(6,6))
    for ii in np.arange(36):
        plt.subplot(6,6,ii+1)
        plt.imshow(images_noise[ii,...])
        plt.axis('off')
    plt.show()
    plt.figure(figsize=(6,6))

    plt.title('Figure 8-9-b Images with Noise')
    plt.savefig('./Images/8_9_b_Images_with-noise.png')

def visualImagesCmp(imor, imnose, imde):
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(imor)
    plt.axis('off')
    plt.title('Origin image')

    plt.subplot(1,3,2)
    plt.imshow(imnose)
    plt.axis('off')
    plt.title('Noise image $\sigma$ = 30')

    plt.subplot(1,3,3)
    plt.imshow(imde)
    plt.axis('off')
    plt.title('De-noise image')

    plt.savefig('./Images/8_11_Visualization-of-images-before-and-after-adding-noise.png')



data_path = 'data/image/stl10_binary/train_X.bin'

# convert bin file to image data
images = read_image(data_path)
print('images.shape: ', images.shape)

# add Gaussian noise on data
images_noise = gaussian_noise(images,30)
print('image_noise: ',images_noise.min(), '-',images_noise.max())

# visualize partial images without noise
# visualImagesWithoutAndWithNoise(images,images_noise)

# data preprocessing
data_Y = np.transpose(images,(0,3,2,1))
data_X = np.transpose(images_noise,(0,3,2,1))

X_train, X_val, y_train, y_val = train_test_split(
    data_X, data_Y, test_size= 0.2, random_state=123
)


X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
print(y_val)
train_data = Data.TensorDataset(X_train,y_train)
val_data = Data.TensorDataset(X_val, y_val)
print('X_train.shape: ', X_train.shape)
print('y_train.shape: ',y_train.shape)
print('X_val.shape: ', X_val.shape)
print('y_val.shape: ',y_val.shape)

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=32,
    shuffle=True,
    # num_workers=4
)

val_loader = Data.DataLoader(
    dataset=val_data,
    batch_size=32,
    shuffle=True,
    # num_workers=4
)

# construct convolutional auto-encoding network
class DenoiseAutoEncoder(nn.Module):
    def __init__(self):
        super(DenoiseAutoEncoder, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=63,
                      kernel_size=3,stride=1,padding=1), # [,64,96,96]
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64,64,3,1,1), # [,64,96,96]
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, 1, 1),  # [,64,96,96]
            nn.ReLU(),
            nn.MaxPool2d(2,2), # [,64,48,48]
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 3, 1, 1),  # [,128,48,48]
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, 3, 1, 1),  # [,128,48,48]
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, 3, 1, 1),  # [,256,48,48]
            nn.ReLU(),
            nn.MaxPool2d(2,2), # [,256,24,24]
            nn.BatchNorm2d(64),

        )

        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(256,128,3,1,1), #[,256,24,24]
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 128, 3, 2, 1,1),  # [,128,48,48]
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, 3, 1, 1),  # [,64,48,48]
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, 3, 1, 1),  # [,32,48,48]
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(32, 32, 3, 1, 1),  # [,32,28,28]
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1), # [,16,96,96]
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.ConvTranspose2d(16, 3, 3, 1, 1), # [,3,96,96]
            nn.Sigmoid()
        )

    def forward(self,x):
        encoder = self.Encoder(x)
        decoder = self.Decoder(encoder)
        return encoder,decoder


DAEmodel = DenoiseAutoEncoder()
print(DAEmodel)

# training and prediction based on Transpose Convolutional Auto-encoding Network
LR = 0.0003
optimizer = torch.optim.Adam(DAEmodel.parameters(), lr=LR)
loss_func = nn.MSELoss()

historyl = hl.History()
canvasl = hl.Canvas()
train_num = 0
val_num = 0

for epoch in range(10):
    train_loss_epoch = 0
    val_loss_epoch = 0
    for step, (b_x,b_y) in enumerate(train_loader):
        DAEmodel.train()
        _,output = DAEmodel(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item() * b_x.size(0)
        train_num = train_num + b_x.size(0)

    for step, (b_x, b_y) in enumerate(val_loader):
        DAEmodel.eval()
        _, output = DAEmodel(b_x)
        loss = loss_func(output,b_y)
        val_loss_epoch += loss.item() * b_x.size(0)
        val_num = val_num + b_x.size(0)

    train_loss = train_loss_epoch / train_num
    val_loss = val_loss_epoch / val_num
    historyl.log(epoch,train_loss=train_loss,
                 val_loss=val_loss)
    with canvasl:
        canvasl.draw_plot(historyl['train_loss'],historyl['val_loss'])

# explore the effect of de-noise

# input
imageindex = 1
im = X_val[imageindex,...]
im = im.unsqueeze(0)
imnose = np.transpose(im.data.numpy(),(0,3,2,1))
imnose = imnose[0,...]

# de-noise
DAEmodel.eval()
_,output = DAEmodel(im)
imde = np.transpose(output.data.numpy(),(0,3,2,1))
imde = imde[0,...]

# output
im = y_val[imageindex,...]
imor =im.unsqueeze(0)
imor = np.transpose(imor.data.numpy(),(0,3,2,1))
imor = imor[0,...]

# compute PSNR after de-noise
# print('PSNR after adding noise: ', compare_psnr(imor,imnose),'dB')
# print('PSNR after de-noise: ',compare_psnr(imor, imnose),'dB')

# images visualization
visualImagesCmp(imor,imnose,imde)

# compute mean value of boosting of PSNR after de-noising on the validation set
PSNR_val = []
DAEmodel.eval()
for ii in range(X_val.shape[0]):
    imageindex = ii
    # inut
    im = X_val[imageindex,...]
    im = im.unsqueeze(0)
    imnose = np.transpose(im.data.numpy(),(0,3,2,1))
    imnose = imnose[0,...]

    # de-noise
    _, output = DAEmodel(im)
    imde = np.transpose(output.data.numpy(),(0,3,2,1))
    imde = imde[0,...]

    # output
    im = y_val[imageindex, ...]
    imor = im.unsqueeze(0)
    imor = np.transpose(im.data.numpy(),(0,3,2,1))
    imor = imor[0,...]


# training and prediction based on upper sampling and convolutional decoding

```

# 2. Illustration

## 2.1 Data Preparation

images.shape:  (5000, 96, 96, 3)

image_noise:  0.0 - 1.0

## 2.2 Model Definition Based on Transpose Convolutional Network


![](./Images/8_9_Images-with-Noise.png)

## 2.3 Training and Prediction Based on Transpose Convolutional Network


## 2.4 Model Definition Based on Upper Sampling and Convolutional Decoder

## 2.5 Training and Prediction Based on Upper Sampling and Convolutional Decoder


