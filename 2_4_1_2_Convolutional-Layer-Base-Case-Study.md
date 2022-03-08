# 2.4.1.2 Convolutional Layer: Base Case Study


## 1. Code
```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def visualImage(myimgray):
    plt.figure(figsize=(6, 6))
    plt.imshow(myimgray, cmap=plt.cm.gray)
    plt.axis("off")
    plt.show()


def saveImage(myimgray, token):
    if type(myimgray) == torch.Tensor:
        myimgray = myimgray.numpy()
    myimgray = myimgray.astype(np.uint8)
    myim_new = Image.fromarray(myimgray)
    myim_new.save(f'data/chap2/Lenna_{token}.png')


def convertConv4d(myimgray_t):
    kersize = 5
    ker = torch.ones(kersize, kersize, dtype=torch.float32) * -1
    ker[2, 2] = 24
    ker = ker.reshape((1, 1, kersize, kersize))
    conv2d = nn.Conv2d(1, 2, (kersize, kersize), bias=False)
    conv2d.weight.data[0] = ker
    imconv2dout = conv2d(myimgray_t)
    return imconv2dout


myim = Image.open('data/chap2/Lenna_orig.png')
myimgray = np.array(myim.convert("L"), dtype=np.float32)

visualImage(myimgray)
saveImage(myimgray, 'gray')
imh, imw = myimgray.shape
myimgray_t = torch.from_numpy(myimgray.reshape((1, 1, imh, imw)))
convertConv4d(myimgray_t)
visualImage(myimgray_t)

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

### 2.2 Convolution

Convolution extracts the outline of the grayscale image.

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



