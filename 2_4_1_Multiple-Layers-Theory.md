# 2.4.1 Multiple Layers: Theory

## 1. Convolutional Layer

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
  - Output: $(N, C_{out}, H_{out}, W_{out})$ where $H_{out}=\ \lfloor(H_in+2\times padding[0]-dilation[0]×(kernel\_\size[0]-1)-1)/stride[0] +1\rfloor$ $W_{out}=\ \lfloor(H_in+2\times padding[0]-dilation[0]×(kernel\_\size[0]-1)-1)/stride[1] +1\rfloor$

## 2. Pooling Layer

| Class | Description |
| :---: | :---: |
| torch.nn.MaxPool1d() | Applies a 1D max pooling over an input signal composed of several input planes. |
| torch.nn.MaxPool2d() | Applies a 2D max pooling over... |
| torch.nn.MaxPool3d() | Applies a 3D max pooling over... |
| torch.nn.MaxUnPool1d() | Computes a partial inverse of ```MaxPool1d```. |
| torch.nn.MaxUnPool2d() | Computes a partial inverse of ```MaxPool2d```. |
| torch.nn.MaxUnPool3d() | Computes a partial inverse of ```MaxPool3d```. |
| torch.nn.AvgPool1d() | Applies a 1D average pooling over an input signal composed of several input planes. |
| torch.nn.AvgPool2d() | Applies a 2D average pooling over... |
| torch.nn.AvgPool3d() | Applies a 3D average pooling over... |
| torch.nn.AdaptiveMaxPool1d() | Applies a 1D adaptive max pooling over an input signal composed of several input planes. |
| torch.nn.AdaptiveMaxPool2d() | Applies a 2D adaptive max pooling over... |
| torch.nn.AdaptiveMaxPool3d() | Applies a 3D adaptive max pooling over... |
| torch.nn.AdaptiveAvgPool1d() | Applies a 1D adaptive average pooling over an input signal composed of several input planes. |
| torch.nn.AdaptiveAvgPool2d() | Applies a 2D adaptive average pooling over... |
| torch.nn.AdaptiveAvgPool3d() | Applies a 3D adaptive average pooling over... |

## 3. Activation Function

Here shows images for 4 activation functions.

## 4. Recurrent Layer

| Class | Description |
| :---: | :---: |
| torch.nn.RNN() | Applies a multi-layer Elman RNN with tanh or Relu non-linearity to an input sequence. |
| torch.nn.LSTM() | Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence. |
| torch.nn.GRU() | Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence. |
| torch.nn.RNNCell() | An Elman RNN cell with tanh or ReLU non-linearity. |
| torch.nn.LSTMCell() | A long short-term memory (LSTM) cell. |
| torch.nn.GRUCell() | A gated recurrent unit (GRU) cell. |

## 5. Fully Connected Layer

| Class | Description |
| :---: | :---: |
| torch.nn.Linear() | Applies a linear transformation to the incoming data: $y = xA^T + b$ |

Fully connected layer = linear layer + activation layer
