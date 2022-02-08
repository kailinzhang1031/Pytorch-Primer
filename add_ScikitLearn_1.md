# SciKitLearn

# 1. Introduction



## 2. Dataset

### 2.1 Theory

#### 2.1.3 Bunch

> sklearn.utils.Bunch(**kwargs)

Container object exposing keys as attributes.

Bunch objects are sometimes used as an output for functions and methods. 

They extend dictionaries by enabling values to be accessed by **key**, ```bunch["value_key"]```, 
or by an **attribute**, ```bunch.value_key```.

- Examples
    ```python
    from sklearn.utils import Bunch
    b = Bunch(a=1, b=2)
    b['b']
    
    b.b
    
    b.a = 3
    b['a']
    
    b.c = 6
    b['c']
    ```


### 2.2 Datasets Overview

#### 2.2.1 Boston

> sklearn.datasets.load_boston(*, return_X_y=False)

- Parameters
  - return_X_ybool, default=False
    - If True, returns **(data, target)** instead of a Bunch object.




