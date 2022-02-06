# 2.3 Auto Gradient

## 1. Auto Gradient


## 2. AUTOMATIC DIFFERENTIATION PACKAGE - TORCH.AUTOGRAD

```torch.autograd``` provides classes and functions implementing automatic differentiation 
of arbitrary scalar valued functions. 

We only need to declare Tensor s for which gradients should be computed 
with the ```requires_grad=True``` keyword.
 
### 1.1 Basic Function
**backward**: Computes the sum of gradients of given tensors with respect to graph leaves.

**grad**: Computes and returns the sum of gradients of outputs with respect to the inputs.

### 1.1.1 Example
```python
x = torch.tensor([[1.0, 2.0], [3.0, 4.0],],requires_grad=True)
# default requires_grad = False
y = torch.sum(x**2+2*x+1)
print("x.requires_grad: ", x.requires_grad)
print("y.requires_grad: ", y.requires_grad)
print("x: ", x)
print("y: ", y)

x.requires_grad:  True
y.requires_grad:  True
x:  tensor([[1., 2.],
        [3., 4.]], requires_grad=True)
y:  tensor(54., grad_fn=<SumBackward0>)
```

Compute gradient of y on x:
```python
y.backward()
x_grad = x.grad
print(x_grad)

tensor([[ 4.,  6.],
        [ 8., 10.]])
```

