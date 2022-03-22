# 3.5.2 Parameter Initialization

This checkpoint introduces methods to initialize parameters.

## 1. Code
```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def specificLayer():
    conv1 = torch.nn.Conv2d(3,16,3)

    torch.manual_seed(12)
    torch.nn.init.normal(conv1.weight,mean=0,std=1)

    torch.nn.init.constant(conv1.bias,val=0.1)


def visualConv1(conv1):
    plt.figure(figsize=(8,6))
    print(conv1.weight)
    print(conv1.weight.data.shape)
    plt.hist(conv1.weight.data.numpy().reshape((-1,1)),bins=30)
    plt.show()


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet,self).__init__()
        self.conv1 = nn.Conv2d(3,16,3)
        self.hidden = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
        )
        self.cla = nn.Linear(50,10)

    def forward(self,x):
        x = self.conv1(x)
        x = x.view(x.shape[0],-1)
        x = self.hidden(x)
        output = self.cla(x)


testnet = TestNet()
# print(testnet)

def int_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.normal(m.weight, mean=0, std=0.5)
    if type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, a=-0.1, b=0.1)
        m.bias.data.fill_(0.01)

torch.manual_seed(12)
testnet.apply(int_weights)

print(testnet.conv1.weight)
```

## 2. Illustration

### 2.1 Weight Initialization For A Layer

- Define a convolutional layer to map 3 features to 16 features.
    ```python
    conv1 = torch.nn.Conv2d(3,16,3)
    ```

- Initialize weight in a specific layer.
    - Using normal distribution.
    ```python
    torch.manual_seed(12)
    torch.nn.init.normal(conv1.weight,mean=0,std=1)    
    ```
    - Using specific value like 0.1 .
    ```python
    torch.nn.init.constant(conv1.bias,val=0.1)
    ```
     - Check the parameter.
       - conv1.weight: torch.Size([16, 3, 3, 3])

### 2.2 Weight Instantiation For A Net

- Define a net and instantiate it.
  ```python
  class TestNet(nn.Module):
      def __init__(self):
          super(TestNet,self).__init__()
          self.conv1 = nn.Conv2d(3,16,3)
          self.hidden = nn.Sequential(
              nn.Linear(100, 100),
              nn.ReLU(),
              nn.Linear(100, 100),
              nn.ReLU(),
          )
          self.cla = nn.Linear(50,10)
  
      def forward(self,x):
          x = self.conv1(x)
          x = x.view(x.shape[0],-1)
          x = self.hidden(x)
          output = self.cla(x)
  
  testnet = TestNet()
  ```

  - Structure:
  ```python
  TestNet(
  (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1))
  (hidden): Sequential(
    (0): Linear(in_features=100, out_features=100, bias=True)
    (1): ReLU()
    (2): Linear(in_features=100, out_features=100, bias=True)
    (3): ReLU()
  )
  (cla): Linear(in_features=50, out_features=10, bias=True)
  )
  ```

- Initialize weights.

  Define the function ```int_weights``` and use ```apply``` in testnet.
  ```python
  def int_weights(m):
      if type(m) == nn.Conv2d:
          torch.nn.init.normal(m.weight, mean=0, std=0.5)
      if type(m) == nn.Linear:
          torch.nn.init.uniform(m.weight, a=-0.1, b=0.1)
          m.bias.data.fill_(0.01)
  torch.manual_seed(12)
  testnet.apply(int_weights)

  print(testnet.conv1.weight)
  ```
  
