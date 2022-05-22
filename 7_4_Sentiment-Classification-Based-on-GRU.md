# 7.4 Sentiment Classification Based on GRU


# 1. Code 


# 2. Illustration

## 2.1 Data Preparation

## 2.2 Model Definition

## 2.3 Training and Prediction


Here we visualize the training process.

GRUNet(
  (embeddnig): Embedding(4, 100)
  (gru): GRU(100, 128, batch_first=True)
  (fcl): Sequential(
    (0): Linear(in_features=128, out_features=128, bias=True)
    (1): Dropout(p=0.5, inplace=False)
    (2): ReLU()
    (3): Linear(in_features=128, out_features=2, bias=True)
  )
)