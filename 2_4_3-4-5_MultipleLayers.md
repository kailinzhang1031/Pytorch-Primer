# 2.4.3-4-5 Multiple Layers

## 0. Introduction

Click here for theory on [activation function]().

Click here for theory on [recurrent layer]().

Click here for theory on [fully connected layer]().

## 1. Activation Funtion

Here shows images for 4 activation function.


## 2. Recurrent Layer

| Classes For Layers | Description |
| :---: | :---: |
| torch.nn.RNN() | Applies a multi-layer Elman RNN with tanh or Relu non-linearity to an input sequence. |
| torch.nn.LSTM() | Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence. |
| torch.nn.GRU() | Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence. |
| torch.nn.RNNCell() | An Elman RNN cell with tanh or ReLU non-linearity. |
| torch.nn.LSTMCell() | A long short-term memory (LSTM) cell. |
| torch.nn.GRUCell() | A gated recurrent unit (GRU) cell. |

More information about **recurrent layer** will be discussed in the following context.


## 3. Fully Connect Layer

| Class(es) For Layer(s) | Description |
| :---: | :---: |
| torch.nn.Linear() | Applies a linear transformation to the incoming data: $y = xA^T + b$ |

Fully connect layer = linear layer + activation layer

More information about **fully connect layer** will be discussed in the following context.