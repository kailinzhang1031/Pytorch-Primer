# 2.4.1.1 Convolutional Layer: Preliminary

## 1. Overview

| Class | Description |
| :---: | :---: |
| nn.Conv1d | Applies a 1D convolution over an input signal composed of several input planes. |
| nn.Conv2d | Applies a 2D convolution... |
| nn.Conv3d | Applies a 3D convolution... |
| nn.ConvTranspose1d | Applies a 1D transposed convolution operator over an input image composed of several input planes.| 
| nn.ConvTranspose2d | Applies a 2D transposed convolution operator... |
| nn.ConvTranspose3d | Applies a 3D transposed convolution operator... |

## 2. CONV2D

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

