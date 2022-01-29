### 2.2.2 Scalar Generation

#### (1) ```torch.tensor()```

> torch.tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False) → Tensor

Constructs a tensor with ```data```.
```python
A = torch.tensor([[1.0,1.0],[2,2]])

tensor([[1., 1.],
        [2., 2.]])
```

> Tensor.size(dim=None) → torch.Size or int

Returns the size of the ```self``` tensor.
If dim is not specified, the returned value is a ```torch.Size```, a subclass of tuple. If dim is specified, returns an int holding the size of that dimension.

We can also call ```.shape``` to get it, they all in the same type of ```torch.Size```
```python
>>> A.shape 
torch.Size([2, 2])

>>> A.size() 
torch.Size([2, 2])

>>> A.size(dim=1) 
2
```

#### (2) ```tensor.Tensor()```

Create with pre-existing data.
```python
>>> C = torch.Tensor([1,2,3,4])

C = tensor([1., 2., 3., 4.])
```

To create a tensor with specific size:

```python
>>> D = torch.Tensor(2,3)

D = tensor([[4.5922e-07, 3.4281e-41, 4.6004e-07],
        [3.4281e-41, 4.6001e-07, 3.4281e-41]])
```

To create a tensor with the **same size** (and similar types) as another tensor, 
use ```torch.*_like``` tensor creation ops:

**Torch.random_like()**

Returns a tensor with the same size as input that is 
filled with random numbers from a **uniform distribution** 
on the interval [0, 1)[0,1). 

```torch.rand_like(input)``` is equivalent to 

```torch.rand(input.size(), dtype=input.dtype, layout=input.layout, 
device=input.device)
``` .
```

**Torch.randn_like**

Returns a tensor with the **same size** as input that is 
filled with random numbers from a **normal distribution** with mean 0 and variance 1. 
```torch.randn_like(input)``` is equivalent to 
```torch.randn(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)```.

We also have:

**zeros_like**: Returns a tensor filled with the scalar value 0, with the same size as **input**.

**ones_like**: Returns a tensor filled with the scalar value 1, with the same size as **input**.

**empty_like**: Returns an uninitialized tensor with the same size as **input**.

**full_like**: 	Returns a tensor with the same size as **input** filled with **fill_value**.


### (3) Scalar and Numpy

> torch.as_tensor(data, dtype=None, device=None) → Tensor

Convert the data into a ```torch.Tensor```. If the **data** is already a ```Tensor``` with the same dtype and device, 
no copy will be performed, otherwise a new ```Tensor``` will be returned with computational graph retained 
if data ```Tensor``` has ```requires_grad=True```. 

Similarly, if the data is an ```ndarray``` of the corresponding dtype and the ```device``` is the cpu, 
no copy will be performed.

Parameters:

- **data** (array_like) – Initial data for the tensor. Can be a list, tuple, NumPy ndarray, scalar, and other types.

- **dtype** (```torch.dtype```, optional) – the desired data type of returned tensor. Default: if None, infers data type from data.

- **device** (```torch.device```, optional) – the desired device of returned tensor. Default: if None, uses the current device for the default tensor type (see torch.set_default_tensor_type()). device will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

Example:

From ```numpy.array``` to ```tensor```, here **device** is set to ```cuda```.

```python
a = numpy.array([1, 2, 3])
t = torch.as_tensor(a)
t

tensor([ 1,  2,  3])
```

Copy is performed:

```python
t[0] = -1
a
array([-1,  2,  3])
```
From ```nd.array``` to ```tensor```.
```python
a = numpy.array([1, 2, 3])
t = torch.as_tensor(a, device=torch.device('cuda'))
t
tensor([ 1,  2,  3])
```

Copy is not perfromed:

```python
t[0] = -1
a

array([1,  2,  3])
```

### (4) Generate scalar by using random generator

#### class Generator

> CLASS torch.Generator(device='cpu') → Generator

Creates and returns a generator object that manages the state of the algorithm which produces pseudo random numbers. 
Used as a keyword argument in many ```In-place random sampling``` functions.

##### Basic
###### Parmeters

**device** (torch.device, optional) – the desired device for the generator.

###### Returns

An torch.Generator object.

##### Return type

Generator

##### Example
```python
>>> g_cpu = torch.Generator()
>>> g_cuda = torch.Generator(device='cuda')
```

###### Functions

**get_state() → Tensor**: Returns the Generator state as a torch.ByteTensor.

**initial_seed() → int**: Returns the initial seed for generating random numbers.
**manual_seed(seed) → Generator**:

Sets the seed for generating random numbers. Returns a torch.Generator object. It is recommended to set a large seed, i.e. a number that has a good balance of 0 and 1 bits. 
Avoid having many 0 bits in the seed.

**seed() → int**: Gets a non-deterministic random number from std::random_device or the current time and uses it to seed a Generator.

**set_state(new_state) → void**: Sets the Generator state.
 
Parameters: 

**new_state** (torch.ByteTensor) – The desired state.

Examples:
```python
>>> g_cpu = torch.Generator()
>>> g_cpu_other = torch.Generator()
>>> g_cpu.set_state(g_cpu_other.get_state())
```

### (4)Random with Normal Distribution

> torch.normal(mean, std, *, generator=None, out=None) → Tensor

Returns a tensor of random numbers drawn from separate normal distributions whose **mean and standard deviation** are given.

Parameters:

- **mean** (Tensor) – the tensor of per-element means

- **std** (Tensor) – the tensor of per-element standard deviations

Example:

```python
>>> torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))
tensor([  1.0425,   3.5672,   2.7969,   4.2925,   4.7229,   6.2134,
          8.0505,   8.1408,   9.0563,  10.0566])
```
### (5) Basic Cases
Some standard or popular used functions to generate scalars are showed as following:

**torch.arrange(start, end, step)**:

Returns a 1-D tensor of size (end−start/step)+1 with values from the interval [start, end) taken with common difference step beginning from start.

**torch.linspace(start,end,steps)**:

Creates a one-dimensional tensor of size **steps** whose values are evenly spaced from **start** to **end**, inclusive. That is, the value are:

(start, start+(end-start)/steps-1, ..., start + (steps-2)* (end-start)/steps-1,end)

**Example**:
```python
>>> torch.linspace(3, 10, steps=5)
tensor([  3.0000,   4.7500,   6.5000,   8.2500,  10.0000])

>>> torch.linspace(-10, 10, steps=5)
tensor([-10.,  -5.,   0.,   5.,  10.])

>>> torch.linspace(start=-10, end=10, steps=5)
tensor([-10.,  -5.,   0.,   5.,  10.])

>>> torch.linspace(start=-10, end=10, steps=1)
tensor([-10.])

```




