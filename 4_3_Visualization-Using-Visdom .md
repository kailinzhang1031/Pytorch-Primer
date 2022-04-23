# 4.3 Visualization Using Visdom

# 1. Code

```python
import numpy as np
import torch
import torch.utils.data as Data
import torchvision

from visdom import Visdom

from sklearn.datasets import load_iris

import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'https://127.0.0.1:7890'



def visualScatter(vis):
    vis.scatter(iris_x[:, 0:2], Y=iris_y + 1, win='2D Scatter', opts=dict(title='2D Scatter'))
    vis.scatter(iris_x[:, 0:3], Y=iris_y + 1, win='3D Scatter', opts=dict(title='3D scatter',
                                                                          markersize=4, xlabel='feature1',
                                                                          ylabel='feature2'))


def visualLineChart(vis):
    x = torch.linspace(-6, 6, 100).view(-1, 1)
    sigmoid = torch.nn.Sigmoid()
    sigmoidy = sigmoid(x)
    tanh = torch.nn.Tanh()
    tanhy = tanh(x)
    relu = torch.nn.ReLU()
    reluy = relu(x)

    ploty = torch.cat((sigmoidy, tanhy, reluy), dim=1)
    plotx = torch.cat((x, x, x), dim=1)

    vis.line(X=plotx, Y=ploty, win='Line Chart', env='main',
             opts=dict(dash=np.array(['solid', 'dash', 'dashdot']),
                       legend=['Sigmoid', 'Tanh', 'ReLU'],
                       title='Plot'))


def visualStem(vis):
    x = torch.linspace(-6, 6, 100).view(-1, 1)
    y1 = torch.sin(x)
    y2 = torch.cos(x)

    plotx = torch.cat((y1, y2), dim=1)
    ploty = torch.cat((x, x), dim=1)

    vis.stem(X=plotx, Y=ploty, win='Stem', opts=dict(lengend=['sin', 'cos'],
                                                     title='Stem'))


def visualHeatmap(vis):
    iris_corr = torch.from_numpy(np.corrcoef(iris_x, rowvar=False))
    vis.heatmap(iris_corr, win='Heatmap', env='main',
                opts=dict(rownames=['x1', 'x2', 'x3', 'x4'],
                          columnames=['x1', 'x2', 'x3', 'x4'],
                          title='Heatmap'))


def visualData(data_loader, vis):
    print('Size of train_data_loader: ', len(data_loader))

    for step, (b_x, b_y) in enumerate(data_loader):
        # print('train_data_x.shape: ', b_x.shape)
        # print('train_data_y.shape: ', b_y.shape)
        vis.image(b_x[0, :, :, :], win='An image', opts=dict(title='An image'))
        vis.images(b_x, win='Images in a batch', nrow=16,
                   opts=dict(title='Images in a batch'))
        if step == 0:
            break


def visualText(vis):
    texts = 'Hello Word!'
    vis.text(texts, win='Text', opts=dict(title='Text'))


iris_x, iris_y = load_iris(return_X_y=True)

train_data = torchvision.datasets.MNIST(
    root='data/Image',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False
)

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=128,
    shuffle=True
)

vis = Visdom(env='main',use_incoming_socket=False)

visualScatter(vis)
visualLineChart(vis)
visualStem(vis)
visualHeatmap(vis)
visualData(train_loader,vis)
visualText(vis)
# vis.save(envs=['main'])

```

# 2. Illustration

## 2.1 Data Preparation

Here we use Iris dataset, which includes 3 classes and 150 instances.

Each instance contains 4 features.

Statistic features are showed as following:

iris_x.shape:  (150, 4)

iris_y.shape:  (150,)

## 2.2 Numeral Data

We can use several methods to visualize the distribution of dataset.

- Scatter

![image](Images/4_5_1_Visualization-using-scatter.png)

- Line Chart

![image](Images/4_5_2_Visualization-using-line-chart.png)

- Stem

![image](Images/4_5_3_Visualization-using-stem.png)

- Heatmap

![image](Images/4_5_4_Visualization-using-heatmap.png)

## 2.3 Image Data

Similarly, we can visualize image data from MNIST dataset.

![image](Images/4_6_1_Visualization-of-image.png )

## 2.2.3 Text Data

We can also visualize text data.

![image](Images/4_7_Visualization-of-text.png
