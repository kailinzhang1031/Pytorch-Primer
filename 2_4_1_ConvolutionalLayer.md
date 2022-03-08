# 2.4.1. Convolutional Layer

### 1.1 Overview

| Class | Description |
| :---: | :---: |
| nn.Conv1d | Applies a 1D convolution over an input signal composed of several input planes. |
| nn.Conv2d | Applies a 2D convolution... |
| nn.Conv3d | Applies a 3D convolution... |
| nn.ConvTranspose1d | Applies a 1D transposed convolution operator over an input image composed of several input planes.| 
| nn.ConvTranspose2d | Applies a 2D transposed convolution operator... |
| nn.ConvTranspose3d | Applies a 3D transposed convolution operator... |

### 1.2 CONV2D
> CLASS torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

- Parameters
  - **in_channels** (int) – Number of channels in the input image

  - **out_channels** (int) – Number of channels produced by the convolution

  - **kernel_size** (int or tuple) – Size of the convolving kernel

  - **stride** (int or tuple, optional) – Stride of the convolution. Default: 1

  - **padding** (int, tuple or str, optional) – Padding added to all four sides of the input. Default: 0

  - **padding_mode** (string, optional) – 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'

  - **dilation** (int or tuple, optional) – Spacing between kernel elements. Default: 1

  - **groups** (int, optional) – Number of blocked connections from input channels to output channels. Default: 1

  - **bias** (bool, optional) – If True, adds a learnable bias to the output. Default: True
  
- Shape
  - Input: $(N, C_{in}, H_{in}, W_{in}) )$
  - Output: $(N, C_{out}, H_{out}, W_{out})$ where $H_out=\ \lfloor(H_in+2\times padding[0]-dilation[0]×(kernel`size[0]-1)-1)/stride[0] +1\rfloor$ $W_out=\ \lfloor(H_in+2\times padding[0]-dilation[0]×(kernel`size[0]-1)-1)/stride[1] +1\rfloor$

## 1.3 Example of Convolutional Layer

### 1.3.1 Data Preprocessing

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
print(myimgray_t.shape)

torch.Size([1, 1, 512, 512])
```

### 1.3.2 Convolution

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
