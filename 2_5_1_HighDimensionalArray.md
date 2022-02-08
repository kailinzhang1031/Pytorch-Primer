# 2.5.1 High Dimensional Array

## 0. Introduction

Overview of tools for data preprocessing.

| Class | Description |
| :---: | :---: |
| torch.utils.data.TensorDataset() | Dataset wrapping tensors. <br>Each sample will be retrieved by indexing tensors along the first dimension. |
| torch.utils.data.ConcatDataset() | Dataset as a concatenation of multiple datasets. <br>This class is useful to assemble different existing datasets. |
| torch.utils.data.Subset() | Subset of a dataset at specified indices. |
| torch.utils.data.Dataloader() | Data loader. Combines **a dataset and a sampler**, and provides an iterable over the given dataset. |
| torch.utils.data.random_split() | Randomly split a dataset into non-overlapping new datasets of given lengths. <br>Optionally fix the generator for reproducible results. |

## 1. Boston for Regression

### 1.1 Data Preprocessing

#### 1.1.1 Data Distribution

```python
import torch
import torch.utils.data as Data
from  sklearn.datasets import  load_boston, load_iris

boston_x, boston_y = load_boston(return_X_y=True)
```
Structure:
- ```(boston_x, boston_y)``` is in the structure of tuple **(data, target)**.
- data: ndarray of shape (506, 13): The data matrix.
- target: ndarray of shape (506,): The regression target.

For details in loading the dataset, please click here.

#### 1.1.2 Type Casting

Convert type of data from **numpy.float64** to **tensor.float32**.
```python
train_xt = torch.from_numpy(boston_x.astype(np.float32))
train_yt = torch.from_numpy(boston_y.astype(np.float32))
```

train_xt.dtype:  torch.float32

train_yt.dtype:  torch.float32



### 1.2 Dataset Construction

#### 1.2.1 Construction

Merge dataset by using ```torch.utils.data.TensorDataset()``` , which includes feature tensor and target tensor.

Note that  **tensor may have the same size of the first dimension**.

Instantiate a **data loader** by using ```torch.utils.data.Dataloader()```.

```python
    train_data = Data.TensorDataset(train_xt,train_yt)

    train_loader = Data.DataLoader(dataset=train_data,
                                   batch_size=64,
                                   shuffle=True)
```

#### 1.2.2 Sample Analysis

Check the dimension of samples in a batch of ```dataloder```:
```python
for step, (b_x,b_y) in enumerate(train_loader):
    if step>0:
        break
```

Structure: 

- ```b_x```: torch.Tensor of torch.Size([64, 13])

- ```b_y```: torch.Tensor of torch.Size([64])

## 2. Iris for Classification

### 2.1 Data Preprocessing

#### 2.1.1 Data Distribution

Load the dataset.

```python
iris_x, iris_y = load_iris(return_X_y=True)
```

#### 2.1.2 Type Casting

Feature label ```iris_x``` is **numpy.ndrray** of **float64**.

Predictive label ```iris_y``` is **numpy.ndarray** of  **int32**.

Convert ```iris_x``` to  **torch.tensor** of **torch.float32**.

Convert ```iris_y``` to **torch.tensor**  of **torch.int64**.

```python
train_xt = torch.from_numpy(iris_x.astype(np.float64))
train_yt = torch.from_numpy(iris_y.astype(np.int64))
```

### 2.2 Dataset Construction

### 2.2.1 Construction

Merge dataset by using ```torch.utils.data.TensorDataset()``` , which includes feature tensor and target tensor.

Instantiate a **data loader** by using ```torch.utils.data.Dataloader()```.

```python
train_data = Data.TensorDataset(train_xt,train_yt)
train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=10,
                               shuffle=True)
```

### 2.2.2 Sample Analysis

Check the dimension of samples in a batch of ```dataloder```.
```python
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
```

Structure:

- ```b_x```: torch.Tensor of torch.Size([10, 4])

- ```b_y```: torch.Tensor of torch.Size([10])

















