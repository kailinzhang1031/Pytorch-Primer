# 2.5.1.2 Boston House Price Forecast: A Case Study

## 1. Code

```python
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.utils.data as Data
from torch.utils.data import DataLoader

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt


def visualY(boston_y):
    plt.figure()
    plt.hist(boston_y, bins=20)
    plt.show()


def checkDataloder(train_loader):
    for step, (b_x, b_y) in enumerate(train_loader):
        print(b_x, b_y)
        if step == 10: break


def visualLoss(train_loss_all):
    plt.plot(train_loss_all, 'r-')
    plt.title('Train loss per iteration')
    plt.show()


boston_X, boston_y = load_boston(return_X_y=True)

visualY(boston_y)

ss = StandardScaler(with_mean=True, with_std=True)
boston_Xs = ss.fit_transform(boston_X)

train_xt = torch.from_numpy(boston_X.astype(np.float32))
train_yt = torch.from_numpy(boston_y.astype(np.float32))
train_data = Data.TensorDataset(train_xt, train_yt)

train_loader = DataLoader(dataset=train_data,
                          batch_size=64,
                          shuffle=True,
                          # num_workers=1,
                          )

checkDataloder(train_loader)


# Defining a model in a basic way
class MLPmodel(nn.Module):
    def __init__(self):
        super(MLPmodel, self).__init__()
        # 1st hidden layer
        self.hidden1 = nn.Linear(
            in_features=13,
            out_features=10,
            bias=True
        )
        self.activate1 = nn.ReLU()
        # 2nd hidden layer
        self.hidden2 = nn.Linear(10, 10)
        self.activate2 = nn.ReLU()
        # Predict and regression layer
        self.regression = nn.Linear(10, 1)

    # Forward propagation
    def forward(self, x):
        x = self.hidden1(x)
        x = self.activate1(x)
        x = self.hidden2(x)
        x = self.activate2(x)
        output = self.regression(x)
        return output


# Concat all layers when defining the model

class MLPmodel2(nn.Module):
    def __init__(self):
        super(MLPmodel2, self).__init__()
        # Hidden layer
        self.hidden = nn.Sequential(
            nn.Linear(13, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
        )
        # Predict and regression layer
        self.regression = nn.Linear(10, 1)

    # Forward propagation
    def forward(self, x):
        x = self.hidden(x)
        output = self.regression(x)
        return output


# mlp = MLPmodel()
mlp = MLPmodel2()
print(mlp)

optimizer = SGD(mlp.parameters(), lr=0.001)
loss_func = nn.MSELoss()
train_loss_all = []

# Training
for epoch in range(30):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = mlp(b_x).flatten()
        train_loss = loss_func(output, b_y)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_loss_all.append(train_loss.item())
print(train_loss_all)

# Visualize loss in every iteration
visualLoss(train_loss_all)

# Saving the model
torch.save(mlp, 'data/chap3/mlp.pkl')

# Loading the model
mlpLoad = torch.load('data/chap3/mlp.pkl')
print(mlpLoad)

# Saving parameters of the model only
torch.save(mlp.state_dict(), 'data/chap3/mlp_param.pkl')

# Loading parameters of the model
mlpParam = torch.load('data/chape3/mlp_param.pkl')
print(mlpParam)

```

## 2. Illustration

### 2.1 Data Preprocessing

#### 2.1.1 Data Distribution

```python
import torch
import torch.utils.data as Data
from torch.utils.data import  DataLoader

import numpy as np

from  sklearn.datasets import load_boston

boston_X, boston_y = load_boston(return_X_y=True)
```


Structure:
- ```(boston_x, boston_y)``` is in the structure of tuple **(data, target)**.
- data: ndarray of shape (506, 13): The data matrix.
- target: ndarray of shape (506,): The regression target.


#### 2.1.2 Type Casting

```python
train_xt = torch.from_numpy(boston_X.astype(np.float32))
train_yt = torch.from_numpy(boston_y.astype(np.float32))
```


train_xt.dtype:  **torch.float32**

train_yt.dtype:  **torch.float32**

#### 2.1.3 Dataset Construction

Merge dataset by using ```torch.utils.data.TensorDataset()``` , which includes feature tensor and target tensor.

Note that  **tensor may have the same size of the first dimension**.

Instantiate a **data loader** by using ```torch.utils.data.Dataloader()```.

```python
train_data = Data.TensorDataset(train_xt, train_yt)

train_loader = DataLoader(dataset=train_data,
                          batch_size=64,
                          shuffle=True)
```

#### 2.1.4 Sample Analysis

Check the dimension of samples in a batch of ```dataloader```:

```python
for step, (b_x, b_y) in enumerate(train_loader):
    print(b_x,b_y)
    if step == 10: break
```

Structure: 

- ```b_x```: torch.Tensor of torch.Size([64, 13])

- ```b_y```: torch.Tensor of torch.Size([64])




### 2.2 Model Definition

Please check this part on code.

### 2.3 Training and Saving

Please check this part on code.