# 2.5.2 Image

# 0. Introduction

Overview of datasets in **torchvision**.

| Class for Dataset | Description |
| :---: | :--- : |
| datasets.MNIST() | [Manual script]() |
| datasets.FashionMNIST() | [Clothes and other 10 classes]() |
| datasets.KMNIST() | [Gray scale data of text]() |
| datasets.CocoCaptions() | [MS COCO for image annotation]() |
| datasets.CocoDetection() | [MS COCO for detection]() |
| datasets.LSUN() | [Classification of 10 Scene and 20 objects]() |
| datasets.CIFAR10() | [CIFAR of 10 classes]() |
| datasets.CIFAR100() | [CIFAR of 100 classes]() |
| datasets.STL10() | [Data of classification for 10 classes and lots of unlabeled]() | 
| datasets.ImageFolder() | [Define a dataset and load data from folder]() |


Overview of image operations in **transform** module of torchvision.

| Class for Operations | Description |
| :----: | :----: |
| transforms.Compose() | Composes several transforms together.   |
| transforms.Scale() | This transform is deprecated in favor of Resize.  |
| transforms.CenterCrop() | Crops the given image at the center.   |
| transforms.RandomCrop() | Crop the given image at a random location.  |
| transforms.RandomHorizontalFlip() | Horizontally flip the given image randomly with a given probability.  |
| transforms.RandomSizedCrop() | This transform is deprecated in favor of RandomResizedCrop.  |
| transforms.Pad() | Pad the given image on all sides with the given “pad” value.  |
| transforms.ToTensor() | Convert a **PIL Image** or **numpy.ndarray** to tensor.<br>This transform does not support torchscript.  |
| transforms.Normalize() | Normalize a tensor image with mean and standard deviation.<br>This transform does not support PIL Image.  |
| transforms.Lambda(lambd) | Apply a user-defined lambda as a transform.<br>This transform does not support torchscript.  |


# 1. Data Preprocessing

## 1.1 Load dataset for training

Load dataset from FashionMNIST.

Convert data to tensor.

Instantiate dataloader to split data into batches.

Number of batches: 938.

```python
import torch
import torch.utils.data as Data
from  torchvision.datasets import  FashionMNIST
import torchvision.transforms as transforms
from  torchvision.datasets import  ImageFolder

train_data  = FashionMNIST(root='./data/Image',
                           train=True,
                           transform=transforms.ToTensor(),
                           download=False)

train_loader = Data.DataLoader(
    dataset= train_data,
    batch_size=64,
    shuffle=True
)
```


## 1.2 Load Data for Testing 

Loading data for testing.

Adding a channel dimension for data.

Scaling the range to (0~1).

```python
test_data = FashionMNIST(root='./data/Image',
                         train=False,
                         download=False)

test_data_x = test_data.data.type(torch.FloatTensor)/255.0
test_data_x = torch.unsqueeze(test_data_x,dim=1)
test_data_y = test_data.targets
```

Shape:

- ```test_data_x```: torch.Size([10000, 1, 28, 28])
- ```test_data_y```: torch.Size([10000])


## 1.3 Preprocessing

Using transforms.Compose() to compose several operations together.
  - transforms.RandomSizedCrop(224)
  - transforms.RandomHorizontalFlip() with p=0.5
  - transforms.ToTensor()
  - transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.226])

```python
    train_data_transforms = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.226])
    ])

```

Using ImageFolder to load data from directories.


Check:
```python
    for step, (b_x, b_y) in enumerate(train_data_loader):
        if step > 0:
            break
```

After transformation:

- Sample ```b_x```: shape of torch.Size([224, 224])
- Sample ```b_y```: shape of torch.Size([4])
- Numeral range: tensor(-2.0182) to tensor(2.4286)





