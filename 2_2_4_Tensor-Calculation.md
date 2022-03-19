# 2.2.4 Tensor Calculation

## 1. Numeral Comparison

### 1.1 Overview
    
| Function | Description |
| :----: | :----: |
| torch.allclose() | close |
| torch.eq() | = |
| torch.ge() | \>= |
| torch.gt() | \> |
| torch.le() | <= |
| torch.lt() | < |
| torch.ne() | != |
| torch.isnan() | NaN |

### 1.2 Torch.allclose

> torch.allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False) → bool

This function checks if all ```input``` and ```other``` satisfy the condition:

$∣input−other∣≤atol+rtol×∣other∣$ elementwise, for all elements of input and other.

The usage of this function is similar to ```numpy.allclose```.

- Parameters
  - input (Tensor) – first tensor to compare
  - other (Tensor) – second tensor to compare
  - atol (float, optional) – absolute tolerance. Default: 1e-08
  - rtol (float, optional) – relative tolerance. Default: 1e-05
  - equal_nan (bool, optional) – if True, then two NaN s will be considered equal. Default: False

- Example

  ```python
  torch.allclose(torch.tensor([10000., 1e-07]), torch.tensor([10000.1, 1e-08]))
  False
  
  torch.allclose(torch.tensor([10000., 1e-08]), torch.tensor([10000.1, 1e-09]))
  True
  
  torch.allclose(torch.tensor([1.0, float('nan')]), torch.tensor([1.0, float('nan')]))
  False
  
  torch.allclose(torch.tensor([1.0, float('nan')]), torch.tensor([1.0, float('nan')]), equal_nan=True)
  True
  ```

## 2. Basic Calculation

### 2.1 Elements Calculation

- addition ```A + B```
- minus ```A - B```
- multiplication ```A * B```
- division ```A / B```
- power ```torch.pow(A)```
- exponential ```torch.exp(A)```
- logarithmic ```torch.log(A)```
- square root ```torch.sqrt(A)```
- reciprocal of square root ```torch.rsqrt(A)```
- max value clamping ```torch.clamp_max(A)```
- min value clamping ```torch.clamp_min(A)```
- range clamping ```torch.clamp(A)```


### 2.2 Matrices Calculation
- transpose 
    ```python
    C = torch.t(A)
    ```

- multiplication 
  ```python
    A.matmul(C)
  ``` 
  or 
  ```python
    A.mm(C)
  ```

- multiplication with specified dimension 
  ```python
    AB[0].eq(torch.matmul(A[0],B[0]))
  ```
- inverse 
  ``` python
  D = torch.inverse(C)
  ```
- trace 
    ```python
    torch.trace(torch.arrange(9.0).reshape(3,3))
    ```
### 2.3 Statistic Calculation

| Function | Description |
| :---: | :---: |
| torch.max() | max value |
| torch.argmax() | position of max value |
| torch.min() | min value |
| torch.argmin() | position of min value |
| torch.sort(A) | sort 1-D tensor |
| torch.sort(A, descending=True) | sort tensor in descending |
| Bsort, Bsort_id(B) = torch.sort(B) | sort 2-D tensor |
| torch.topk() | Returns the k largest elements of the given input tensor along a given dimension.|
| torch.kthvalue() | Returns a namedtuple ```(values, indices)``` where ```values``` is the ```k``` th smallest element 
| | of each row of the input tensor in the given dimension dim. And indices is the index location of each element found.|
| torch.mean() | Returns the mean value of all elements in the input tensor. |
| torch.sum() | Returns the sum of all elements in the input tensor. |
| torch.cumsum() | Returns the cumulative sum of elements of input in the dimension ```dim```. |
| torch.median() | Returns the median of the values in input. |
| torch.std() | Calculates the standard deviation of all elements in the input tensor. |
| | unbiased (bool) – whether to use Bessel’s correction $(\delta N = 1)$. 
| | If unbiased is True, Bessel’s correction will be used. Otherwise, the sample deviation is calculated, without any correction.
| | dim (int or tuple of python:ints) – the dimension or dimensions to reduce.
