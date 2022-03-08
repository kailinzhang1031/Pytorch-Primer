# 2.2.1 Data Types Of Tensor

##  1.  Initializing and Basic Operations

### 1.1 Construction

A tensor can be constructed from a Python list or sequence using the ``torch.tensor()`` **constructor**:

```python
torch.tensor([[1., -1.], [1., -1.]])

tensor([[ 1.0000, -1.0000],
        [ 1.0000, -1.0000]])

torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))

tensor([[ 1,  2,  3],
        [ 4,  5,  6]])
```

### 1.2 Data Type

A tensor of specific data type can be constructed by passing a ```torch.dtype``` and/or a ```torch.device``` 
to a **constructor** or **tensor creation op**:

```python
torch.zeros([2, 4], dtype=torch.int32)
tensor([[ 0,  0,  0,  0],
        [ 0,  0,  0,  0]], dtype=torch.int32)

cuda0 = torch.device('cuda:0')

torch.ones([2, 4], dtype=torch.float64, device=cuda0)
tensor([[ 1.0000,  1.0000,  1.0000,  1.0000],
        [ 1.0000,  1.0000,  1.0000,  1.0000]], dtype=torch.float64, device='cuda:0')

```
#### 1.2.1 Basic Data Types

> Class Torch.Tensor

A **torch.Tensor** is a multi-dimensional matrix containing elements of a single data type<sup>[1]</sup>.

Torch defines 10 tensor types with CPU and GPU variants which are as follows.

##### 1.2.2 Data Type Conversion

```Tensor.int()``` is a function used to transfer data between different datatypes.
```self.int()``` is equivalent to ```self.to(torch.int32)```.

```python
a = torch.tensor([1.2,3.4])

a.dtype:  torch.float32
a.int():  tensor([1, 3], dtype=torch.int32)
a.float():  tensor([1.2000, 3.4000])
a.to(torch.int32):  tensor([1, 3], dtype=torch.int32)
```

##### 1.2.2 Set Default Data Type
```python
torch.set_default_dtype(d)
```
Sets the default floating point dtype to ```d```. 

Initial default for floating point is ```torch.float32```, Python floats are interpreted as float32.
```python
torch.tensor([1.2, 3]).dtype

torch.float32

torch.set_default_dtype(torch.float64)

```
Python floats are now interpreted as float64:
```python
torch.tensor([1.2, 3]).dtype    # a new floating point tensor

torch.float64
```

##### 1.2.3 Get Default Data Type

> torch.get_default_dtype() → torch.dtype

Get the current default floating point ```torch.dtype```.

- Examples:

Initial default for floating point is ```torch.float32```:
```python
torch.get_default_dtype()

torch.float32
```

```python
torch.set_default_dtype(torch.float64)
```
Default is now changed to ```torch.float64```:
```python
torch.get_default_dtype()

torch.float64
```
Setting tensor type also affects this:
```python
torch.set_default_tensor_type(torch.FloatTensor)
```
Changed to ```torch.float32```, the dtype for **torch.FloatTensor**:
```python
torch.get_default_dtype()

torch.float32
```
