# 2.4.1.2 Multiple Layers: Base Case Study

## 1. Code
```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def convertConv4d(myimgray):
    print('Size before convolution: ', myimgray.shape)
    kersize = 5
    ker = torch.ones(kersize, kersize, dtype=torch.float32) * -1
    ker[2, 2] = 24
    ker = ker.reshape((1, 1, kersize, kersize))
    conv2d = nn.Conv2d(1, 2, (kersize, kersize), bias=False)
    conv2d.weight.data[0] = ker
    imconv2dout = conv2d(myimgray_t)
    imconv2dout_im = imconv2dout.data.squeeze()
    print('Size after convolution: ', imconv2dout_im.shape)
    # visualConv4d(imconv2dout_im)
    return imconv2dout


def maxPool(imconv2dout):
    print('Size before MaxPooling: ', imconv2dout.shape)
    kernel_size = 5
    torch.nn.MaxPool2d(kernel_size=kernel_size, stride=None, padding=0, dilation=1, return_indices=False,
                       ceil_mode=False)
    maxpool2 = nn.MaxPool2d(2, stride=2)
    pool2_out = maxpool2(imconv2dout)
    pool2_out_im = pool2_out.squeeze()
    print('Size after MaxPooling: ', pool2_out_im.shape)
    return pool2_out_im


def avgPool(imconv2out):
    print('Size before AveragePooling: ', imconv2out.shape)
    avgPool2 = nn.AvgPool2d(2, stride=2)
    pool2_out = avgPool2(imconv2out)
    pool2_out_im = pool2_out.squeeze()
    print('Size after AveragePooling:', pool2_out.shape)
    return pool2_out_im


def visualConv4d(imconv2dout_im):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(imconv2dout_im[0], cmap=plt.cm.gray)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(imconv2dout_im[1], cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()
    # saveImage(imconv2dout_im[0],token='outline')
    # saveImage(imconv2dout_im[1],token='random')


def visualMaxAvgPool(pool2_out_im):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(pool2_out_im[0].data, cmap=plt.cm.gray)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(pool2_out_im[1].data, cmap=plt.cm.gray)
    plt.axis("off")
    plt.show()


def saveImage(myimgray, token):
    if type(myimgray) == torch.Tensor:
        myimgray = myimgray.numpy()
    myimgray = myimgray.astype(np.uint8)
    myim_new = Image.fromarray(myimgray)
    myim_new.save(f'data/chap2/Lenna_{token}.png')


def diff(myimgray):
    im = np.array(myimgray)
    im = np.expand_dims(im, axis=2)
    print(im.shape)
    imnew = im - myim
    imnew = imnew.astype(np.uint8)
    print(imnew)
    plt.imshow(imnew)
    plt.show()


def visualImage():
    plt.figure(figsize=(6, 6))
    plt.imshow(myimgray, cmap=plt.cm.gray)
    plt.axis("off")
    plt.show()


myim = Image.open('data/chap2/Lenna_orig.png')
myimgray = np.array(myim.convert("L"), dtype=np.float32)

imh, imw = myimgray.shape
myimgray_t = torch.from_numpy(myimgray.reshape((1, 1, imh, imw)))
# print(myimgray_t.shape)
# print(myimgray_t)
# convertConv4d(myimgray)


imconv2dout = convertConv4d(myimgray)
visualConv4d(myimgray)

pool2_out_im_1 = maxPool(imconv2dout)
visualMaxAvgPool(pool2_out_im_1)

pool2_out_im_2 = avgPool(imconv2dout)
visualMaxAvgPool(pool2_out_im_2)

```

## 2. Illustration

### 2.1 Data Preprocessing

**(1) Module import, environment initialization**

  ```python
  import torch
  import torch.nn as nn
  import matplotlib.pyplot as plt
  from PIL import Image
  import numpy as np
  
  import os
  os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
  ```

**(2) Loading and conversion**

Load the image, convert the image to array in gray style as array.

```python
myim = Image.open('../data/chap2/Lenna_orig.png')
myimgray = np.array(myim.convert("L"), dtype=np.float32)
```

Convert ```myim``` to ```numpy.nparray```,  ```myim``` is
a matrix in the shape of **(512, 512, 3)**, while ```myimgray```
is a matrix in the shape of **(512, 512)**, which is in **gray style**.

- Visualization
  ```python
  plt.figure(figsize=(6, 6))
  plt.imshow(myimgray, cmap=plt.cm.gray)
  plt.axis("off")
  plt.show()
  ```

- Saving
  ```python
  def saveImage(myimgray, token):
      if type(myimgray) == torch.Tensor:
          myimgray = myimgray.numpy()
      myimgray = myimgray.astype(np.uint8)
      myim_new = Image.fromarray(myimgray)
      myim_new.save(f'data/chap2/Lenna_{token}.png')
  
  saveImage(myimgray,gray)
  ```

**(3) Ndarray to Tensor**

Reshape the image array in the size of **(512,512)** to **(1,1,512,512)**.

Convert the image array to tensor.

```python
imh, imw = myimgray.shape
myimgray_t = torch.from_numpy(myimgray.reshape((1, 1, imh, imw)))

torch.Size([1, 1, 512, 512])
```

### 2.2 Convolutional Layer

Convolutional Layer extracts the outline of the grayscale image.

Define edge detection convolutional kernels and set dimensions as 1 * 1 * 5 * 5.

```python
def convertConv4d(myimgray):
    kersize = 5
    ker = torch.ones(kersize, kersize, dtype=torch.float32) * -1
    ker[2, 2] = 24
    ker = ker.reshape((1, 1, kersize, kersize))
```

Convolution operation.
```python
    conv2d = nn.Conv2d(1, 2, (kersize, kersize), bias=False)
```

Set kernels in convolution, using edge detection kernel at first.
```python
    conv2d.weight.data[0] = ker
```

Convolution operations on grayscale image.
```python
    imconv2dout = conv2d(myimgray_t)
```

Squeezing the dimension of the output after convolution.
```python
    imconv2dout_im = imconv2dout.data.squeeze()
    print('Size after convolution: ', imconv2dout_im.shape)

Size after convolution:  torch.Size([2, 508, 508])
```

Here we show 3 images after convolution.

Kernel which extracts the outline detection finely gets the outline information of the image.

Kernel using random numbers shows differently according to the exact number.

### 2.3 Max Pooling Layer

Size before convolution:  **(512, 512)**

Size after convolution:  **torch.Size([2, 508, 508])**

Size before MaxPooling:  **torch.Size([1, 2, 508, 508])**

Size after MaxPooling:  **torch.Size([2, 254, 254])**

### 2.4 Average Pooling Layer

Size before convolution:  (512, 512)

Size after convolution:  torch.Size([2, 508, 508])

Size before AveragePooling:  torch.Size([1, 2, 508, 508])

Size after AveragePooling: torch.Size([1, 2, 254, 254])