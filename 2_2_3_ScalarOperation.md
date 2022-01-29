# 2.2.3 Scalar Operation

## 1. Reshape

### 1.1 Torch.reshape

> torch.reshape(input, shape) → Tensor

Returns a tensor with the same data and number of elements as input, but with the specified shape. 
When possible, the returned tensor will be a view of input. 
Otherwise, it will be a copy. 

#### Parameters

- **input** (Tensor) – the tensor to be reshaped

- **shape** (tuple of python:ints) – the new shape


#### Example
```python
>>> a = torch.arange(4.)
>>> torch.reshape(a, (2, 2))
tensor([[ 0.,  1.],
        [ 2.,  3.]])
```
```python
>>> b = torch.tensor([[0, 1], [2, 3]])
>>> torch.reshape(b, (-1,))
tensor([ 0,  1,  2,  3])

```

torch.reshape() is simillar to torch.view()

For more detailed illustration, please click here.

https://blog.csdn.net/Flag_ing/article/details/109129752

https://stackoverflow.com/questions/49643225/whats-the-difference-between-reshape-and-view-in-pytorch


### 1.2 Torch.tensor.resize_as

> Tensor.resize_as_(tensor, memory_format=torch.contiguous_format) → Tensor

Resizes the ```self``` tensor to be the same size as the specified ```tensor```. 
This is equivalent to ```self.resize_(tensor.size())```.

##### Parameter 
- **memory_format** (```torch.memory_format```, optional) – the desired memory format of Tensor. 
Default: torch.contiguous_format.

### 1.3.1 Torch.unsqueeze

> torch.unsqueeze(input, dim) → Tensor

Returns a new tensor with a dimension of size one inserted at the specified position.

The returned tensor shares the same underlying data with this tensor.

A **dim** value within the range **[-input.dim() - 1, input.dim() + 1)** can be used. 
Negative dim will correspond to ```unsqueeze()``` applied at **dim** = **dim** + input.dim() + 1.

```python
>>> x = torch.tensor([1, 2, 3, 4])
>>> torch.unsqueeze(x, 0)

tensor([[ 1,  2,  3,  4]])
```
The shape of x turns from torch.Size([4]) to torch.Size([1, 4])
```python
>>> torch.unsqueeze(x, 1)

tensor([[ 1],
        [ 2],
        [ 3],
        [ 4]])
```
The shape of x turns from torch.Size([4]) to torch.Size([4, 1]).

### 1.3.2 Torch.squeeze

> torch.squeeze(input, dim=None, *, out=None) → Tensor

Returns a tensor with all the dimensions of input of size 1 removed.

For example, if input is of shape: A * 1 * B * C * 1 * D)
then the out tensor will be of shape: (A * B * C * D).

When dim is given, a squeeze operation is done only in the given dimension. If input is of shape: (A * 1 * B), 
```squeeze(input, 0)``` leaves the tensor unchanged, but ```squeeze(input, 1)``` will squeeze the tensor to the shape 
(A * B).

#### Parameters
- input (Tensor) – the input tensor.

- dim (int, optional) – if given, the input will be squeezed only in this dimension

#### Examples

```python
>>> x = torch.zeros(2, 1, 2, 1, 2)
>>> x.size()
torch.Size([2, 1, 2, 1, 2])
>>> y = torch.squeeze(x)
>>> y.size()
torch.Size([2, 2, 2])
>>> y = torch.squeeze(x, 0)
>>> y.size()
torch.Size([2, 1, 2, 1, 2])
>>> y = torch.squeeze(x, 1)
>>> y.size()
torch.Size([2, 2, 1, 2])

```

### 1.4 Torch.tensor.expand

> Tensor.expand(*sizes) → Tensor

Returns a new view of the ```self``` tensor with singleton dimensions expanded to a larger size.

Passing -1 as the size for a dimension means not changing the size of that dimension.

##### Example1
```python
>>> x = torch.tensor([[1], [2], [3]])
>>> x.size()
torch.Size([3, 1])
```
```python
>>> x.expand(3, 4)

tensor([[ 1,  1,  1,  1],
        [ 2,  2,  2,  2],
        [ 3,  3,  3,  3]])
```

```python
>>> x.expand(-1, 4)   # -1 means not changing the size of that dimension
tensor([[ 1,  1,  1,  1],
        [ 2,  2,  2,  2],
        [ 3,  3,  3,  3]])
```

When expanding the dimension of Tensor in higher dimension, the first element of torch.Size must
remain the same, which is the fundamental unit of a tensor.

```python
x = torch.tensor([[1], [2], [3]])

x.size()

torch.Size([3, 1])

x = torch.unsqueeze(x,-1)
x.size()
torch.Size([3, 1, 1])

x = x.expand(3,1,5)
torch.Size([3, 1, 5])

x= x.expand(3,2,-1)

tensor([[[1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1]],

        [[2, 2, 2, 2, 2],
         [2, 2, 2, 2, 2]],

        [[3, 3, 3, 3, 3],
         [3, 3, 3, 3, 3]]])

torch.Size([3, 2, 5])
```

### 1.5 Torch.tensor.repeat

> Tensor.repeat(*sizes) → Tensor
 
Repeats this tensor along the specified dimensions.

Unlike ```expand()```, this function **copies the tensor’s data**.


#### Note

```repeat()``` behaves differently from ```numpy.repeat```, but is more similar to numpy.tile. 
For the operator similar to ```numpy.repeat```, see torch.repeat_interleave().

#### Parameters

**sizes** (torch.Size or int...) – The number of times to repeat this tensor along each dimension

#### Example:

```python
>>> x = torch.tensor([1, 2, 3])
>>> x.repeat(4, 2)
tensor([[ 1,  2,  3,  1,  2,  3],
        [ 1,  2,  3,  1,  2,  3],
        [ 1,  2,  3,  1,  2,  3],
        [ 1,  2,  3,  1,  2,  3]])

>>> x.repeat(4, 2, 1).size()
torch.Size([4, 2, 3])

```

**"copies the tensor’s data"** means when **repeat** a tensor to higher dimension,
the tensor will full-fill every side of the space.

To the opposite, **expand** a tensor means that the tensor will be mapped to a higher
dimension without copy.
```python
>> x = torch.tensor([1,2,3])
>> x.size()
torch.Size([3])

>> x.expand(3,3)
tensor([[1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]])

>> x.size()
torch.Size([3, 3])

```

repeat:
```python
>> x = torch.tensor([1,2,3])
>> x.size()
torch.Size([3])

>> x.repeat(3,3)
tensor([[1, 2, 3, 1, 2, 3, 1, 2, 3],
        [1, 2, 3, 1, 2, 3, 1, 2, 3],
        [1, 2, 3, 1, 2, 3, 1, 2, 3]])

x.size()
torch.Size([3, 9])
```

```python
>> x = torch.tensor([1,2,3])
>> x.size()
torch.Size([3])

>> x.repeat(3,5)
tensor([[1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]])

x.size()
torch.Size([3, 15])
```

The process of repeat goes as copying the data of tensor from low dimension to higher
dimension.

The process of expand goes as mapping the tensor to a higher dimension, which means
that the parameter in the lower dimension muse match the original size.

i.e. following statement is not allowed:
```python
x = torch.tensor([1,2,3])
x = x.expand(3,5)

x = x.expand(3,5)
RuntimeError: The expanded size of the tensor (5) must match the existing size (3) at non-singleton dimension 1.  Target sizes: [3, 5].  Tensor sizes: [3]


```

## 2. Get Elements From Scalar

### 2.1 Slice and index

```python
>> A = torch.arange(12).reshape(1,3,4)

tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]]])

>> A[0]

tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])

>> A[0,0:2,:]

tensor([[0, 1, 2, 3],
        [4, 5, 6, 7]])
```

The slice statement can be interpreted as get elements in _(z,y,x) = (0,[0:2],[:])_.



### 2.2 Conditional selection by torch.where()

> torch.where(condition, x, y) → Tensor

Return a tensor of elements selected from either x or y, depending on condition.

#### Note
The tensors ```condition```, ```x```, ```y``` must be broadcastable.

#### Example
When ```condition``` is relative to ```x``` and ```y```,

```torch.where()``` is similar to ```replace```, which replaces original string by target string.

```torch.where()```  replace the original string in the position where ```condition``` is True.

```python
>>> a = torch.arange(-6,6).reshape(3,4)
tensor([[-6, -5, -4, -3],
        [-2, -1,  0,  1],
        [ 2,  3,  4,  5]])

>> b = torch.arange(100,112).reshape(3,4)
tensor([[100, 101, 102, 103],
        [104, 105, 106, 107],
        [108, 109, 110, 111]])

>> c = torch.where(a>0,a,b)
tensor([[100, 101, 102, 103],
        [104, 105, 106,   1],
        [  2,   3,   4,   5]])
```

### 2.3 Conditional selection by setting index as a boolean value
```python
>>> a = torch.arange(0,10)
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

>>> a[a>5]
tensor([6, 7, 8, 9])
```

### 2.4 Structural selection

```torch.tril(input, diagonal=0, *, out=None) → Tensor```: Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices ```input```, the other elements of the result tensor ```out``` are set to 0.

```torch.triu(input, diagonal=0, *, out=None) → Tensor```: Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices ```input```, the other elements of the result tensor ```out``` are set to 0.

```torch.diag(input, diagonal=0, *, out=None) → Tensor```: 
- If input is **a vector (1-D tensor)**, then returns a 2-D square tensor with the elements of input as the diagonal.

```python
>> a = torch.arange(12)

tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

>>> a.size()

torch.Size([12])

>>> b = torch.diag(a)

tensor([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  3,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  4,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  5,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  6,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  7,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  8,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  9,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 10,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 11]])

>>> b.size()
torch.Size([12, 12])
```

- If input is **a matrix (2-D tensor)**, then returns a 1-D tensor with the diagonal elements of input.
    ```python
    >> a = torch.arange(25).reshape(5,5)
    
    tensor([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]])
    
    >> a.size()
    
    torch.Size([5, 5])
    
    >> b = torch.diag(a)
    
    tensor([ 0,  6, 12, 18, 24])
    
    >> b.size()
    
    torch.Size([5])
    ```

    The argument ```diagonal``` controls which diagonal to consider:

    Here we use **rectangular matrix** for example.

- If ```diagonal``` = 0, it is the main diagonal.

    **Main diagonal** in a rectangular matrix is considered from **the position (0,0)**.

    ```python
    >> a = torch.arange(12).reshape(3,4)
    
    tensor([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]])
    
    >> a.size()
    
    torch.Size([3, 4])
    
    >> b = torch.diag(a)
    
    tensor([ 0,  5, 10])
    
    >> b.size()
    
    torch.Size([3])
    ```


- If ```diagonal``` > 0, it is **above** the main diagonal.
    ```python
    >> b = torch.diag(a,diagonal=1)
    
    tensor([2, 7])
    
    >> b.size()
    
    torch.Size([2])
    ```

- If ```diagonal``` < 0, it is **below** the main diagonal.
    ```python
    >> b = torch.diag(a,diagonal=-1)
    
    tensor([4, 9])
    
    >> b.size()
    
    torch.Size([2])
    ```


## 3. Concatenating and Splitting

### 3.1 Concatenating by torch.cat
> torch.cat(tensors, dim=0, *, out=None) → Tensor

Concatenates the given sequence of seq tensors in the given dimension. 

All tensors must either have the same shape (except in the concatenating dimension) or be empty.

```torch.cat()``` can be seen as an inverse operation for ```torch.split()``` and ```torch.chunk()```.

```torch.cat()``` can be best understood via examples.

#### Parameters

- **tensors** (sequence of Tensors) – any python sequence of tensors of the same type. Non-empty tensors provided must have the same shape, except in the cat dimension.

- **dim** (int, optional) – the dimension over which the tensors are concatenated.

#### Examples
```python
>> a = torch.arange(12).reshape(3,4)

tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])

>> b = torch.arange(100,112).reshape(3,4)

tensor([[100, 101, 102, 103],
        [104, 105, 106, 107],
        [108, 109, 110, 111]])
```
- Concatenating **2 tensors** along **dimension 0**:
    ```python
    >> c = torch.cat((a,b),0)
    
    tensor([[  0,   1,   2,   3],
            [  4,   5,   6,   7],
            [  8,   9,  10,  11],
            [100, 101, 102, 103],
            [104, 105, 106, 107],
            [108, 109, 110, 111]])
    
    c.size()
    
    torch.Size([6, 4])
    ```
- Concatenating **2 tensors** along **dimension 1**:

    ```python
    >> d = torch.cat((a,b),1)
    
    tensor([[  0,   1,   2,   3, 100, 101, 102, 103],
            [  4,   5,   6,   7, 104, 105, 106, 107],
            [  8,   9,  10,  11, 108, 109, 110, 111]])
    
    >> d.size()
    
    torch.Size([3, 8])
    ```

- Concatenating **3 tensors** along **dimension 0**:
    ```python
    >> a = torch.arange(0, 12).reshape(3, 4)
    
    tensor([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]])
    
    >> b = torch.arange(12, 24).reshape(3, 4)
    
    tensor([[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]])
    
    >> c = torch.arange(24, 36).reshape(3, 4)
    
    tensor([[24, 25, 26, 27],
            [28, 29, 30, 31],
            [32, 33, 34, 35]])
    
    >> d = torch.cat((a,b,c),0)
    
    tensor([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11],
            [12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23],
            [24, 25, 26, 27],
            [28, 29, 30, 31],
            [32, 33, 34, 35]])
    
    >> d.size()
    
    torch.Size([9, 4])
    ```

- Concatenating **3 tensors** along **dimension 1**:
    ```python
    >> d = torch.cat((a,b,c),1)
    
    tensor([[ 0,  1,  2,  3, 12, 13, 14, 15, 24, 25, 26, 27],
            [ 4,  5,  6,  7, 16, 17, 18, 19, 28, 29, 30, 31],
            [ 8,  9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35]])
    
    >> d.size()
    
    torch.Size([3, 12])
    ```

### 3.2 Concatenating by torch.stack()
> torch.stack(tensors, dim=0, *, out=None) → Tensor

Concatenates a sequence of tensors along a **new dimension**.

The tensor will be converted from **(A,B)** to **(H,A,B)**.

All tensors need to be of **the same size**.

#### Examples
- Tensors in the **different** shapes:
    ```python
    >> a = torch.arange(0,12).reshape(3,4)
    
    tensor([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]])
    
    >> a.size()
    
    torch.Size([3, 4])
    
    >> b = torch.arange(12,24).reshape(2,6)
    
    tensor([[12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23]])
    
    >> c = torch.stack((a,b))
    
        c = torch.stack((a,b))
    RuntimeError: stack expects each tensor to be equal size, 
    but got [3, 4] at entry 0 and [2, 6] at entry 1
    ```
- Tensors in the **same** shapes:
    ```python
    a = torch.arange(0,12).reshape(3,4)
    
    tensor([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]])
    
    a.size()
    torch.Size([3, 4])
    
    b = torch.arange(12,24).reshape(3,4)
    
    tensor([[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]])
    
    >> b.size()
    torch.Size([3, 4])
    
    >> c = torch.stack((a,b))
    tensor([[[ 0,  1,  2,  3],
             [ 4,  5,  6,  7],
             [ 8,  9, 10, 11]],
    
            [[12, 13, 14, 15],
             [16, 17, 18, 19],
             [20, 21, 22, 23]]])
    >> c.size()
    torch.Size([2, 3, 4])
    ```

    ```a,b``` are concatenated to ```c```, from [3,4] to [2,3,4].

### 3.3 Splitting

> torch.chunk(input, chunks, dim=0) → List of Tensors

Attempts to split a tensor into the specified number of chunks. 

Each chunk is a **view** of the input tensor.

#### Parameters
- **input** (Tensor) – the tensor to split

- **chunks** (int) – number of chunks to return

- **dim** (int) – dimension along which to split the tensor

#### Examples

- Splitting

  - If the tensor size along the given dimesion ```dim``` is divisible by ```chunks```, 
  **all returned chunks will be the same size**. 
    ```python
    >>> torch.arange(11).chunk(6)
    (tensor([0, 1]),
     tensor([2, 3]),
     tensor([4, 5]),
     tensor([6, 7]),
     tensor([8, 9]),
     tensor([10]))
    ```
  - If the tensor size along the given dimension ```dim``` is not divisible by ```chunks```, 
  **all returned chunks will be the same size, except the last one**. 
    ```python
    >>> torch.arange(12).chunk(6)
    (tensor([0, 1]),
     tensor([2, 3]),
     tensor([4, 5]),
     tensor([6, 7]),
     tensor([8, 9]),
     tensor([10, 11]))
    ```

  - If such Splitting is not possible, 
  this function may **return less than the specified number of chunks**.
    ```python
    >>> torch.arange(13).chunk(6)
    (tensor([0, 1, 2]),
     tensor([3, 4, 5]),
     tensor([6, 7, 8]),
     tensor([ 9, 10, 11]),
     tensor([12]))
    ```

- Splitting along dimensions

  - Splitting along **dimension 0**
    ```python
    >>> a = torch.arange(12).reshape(3,4)
    tensor([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]])
    
    >>> b = torch.chunk(a,3,dim=0)
    (tensor([[0, 1, 2, 3]]), tensor([[4, 5, 6, 7]]), tensor([[ 8,  9, 10, 11]]))
    
    >>> a.size()
    torch.Size([3, 4])
    
    >>> b[0]
    tensor([[0, 1, 2, 3]])
    
    >>> b[0].size()
    torch.Size([1, 4])
    ```
  - Splitting along **dimension 1**
    ````python
    >>> a = torch.arange(12).reshape(3,4)
    tensor([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]])
    
    >>> b = torch.chunk(a,2,dim=1)
    (tensor([[0, 1],
            [4, 5],
            [8, 9]]), tensor([[ 2,  3],
            [ 6,  7],
            [10, 11]]))
    
    >>> a.size()
    torch.Size([3, 4])
    
    >>> b[0]
    tensor([[0, 1],
            [4, 5],
            [8, 9]])
    
    >>> b[0].size()
    torch.Size([3, 2])
    ````