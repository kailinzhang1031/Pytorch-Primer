# 5.2 MLP Classification Model

# 1. Code

## 2. Illustration

## 2.1 Data Preprocessing

### 2.1.1 Data Distribution

Use Spam dataset.

Analysis the dataset. Statistics features are shown as following:

- Instance: tuple in the size of (58,1), 58 attributes.
- Total instance: 4601
- Total attributes: 58
  - 48 continuous real [0,100] attributes of type word_freq_WORD 
    <br>= percentage of words in the e-mail that match WORD, i.e. 100 * (number of times the WORD appears in the e-mail) / total number of words in e-mail. A "word" in this case is any string of alphanumeric characters bounded by non-alphanumeric characters or end-of-string.
  - 6 continuous real [0,100] attributes of type char_freq_CHAR] 
    <br>= percentage of characters in the e-mail that match CHAR, i.e. 100 * (number of CHAR occurences) / total characters in e-mail
  - 1 continuous real [1,...] attribute of type capital_run_length_average 
    <br>= average length of uninterrupted sequences of capital letters
  - 1 continuous integer [1,...] attribute of type capital_run_length_longest 
    <br>= length of longest uninterrupted sequence of capital letters 
  - 1 continuous integer [1,...] attribute of type capital_run_length_total 
    <br>= sum of length of uninterrupted sequences of capital letters  
    <br>= total number of capital letters in the e-mail 
  - 1 nominal {0,1} class attribute of type spam 
    <br>= denotes whether the e-mail was considered spam (1) or not (0), i.e. unsolicited commercial e-mail.
  

Statistic on two classes of samples using pd.value_counts().

```python
    word_freq_make  word_freq_address  ...  capital_run_length_total  label
0             0.00               0.64  ...                       278      1
1             0.21               0.28  ...                      1028      1
2             0.06               0.00  ...                      2259      1
3             0.00               0.00  ...                       191      1
4             0.00               0.00  ...                       191      1
..             ...                ...  ...                       ...    ...
95            0.00               0.00  ...                        91      1
96            0.00               0.35  ...                       313      1
97            0.00               0.43  ...                       222      1
98            0.00               0.00  ...                       191      1
99            1.24               0.41  ...                       114      1
```

```python
[100 rows x 58 columns]
0    2788
1    1813
Name: label, dtype: int64
```

Here we get 1813 junk mail samples and 2788 non-junk mail samples.


Purely integer-location based indexing for selection by position.

https://towardsdatascience.com/what-and-why-behind-fit-transform-vs-transform-in-scikit-learn-78f915cf96fe

![5_2_boxplot.png]


### 2.1.2 Dataset Split

Splitting spam dataset by 75% for train set and 25% for test set.

Here we use train_test_split().

### 2.1.3 Standardization

Using MinMaxScaler to standardize data, transforming value of each feature to 0~1.

### 2.1.4 Visualization

Using boxplot to visualize every feature of train dataset.

Comparing data distribution of different classes of mails.

Some features show great differences on different kinds of mails, 
like word_freq_all, word_freq_our, word_freq_your, word_freq)you, word_freq_000, etc.

## 2.2 Model


### 2.2.1 Model Definition & Visualization


### 2.2.2 Training

Training model using unprocessed data.

The result of training and testing on unprocessed data is shown below.

![https://github.com/kailinzhang1031/Pytorch-Primer/blob/main/Images/5_6_Training-Without-Standardization.png]

The loss function is fluctuating and doesn't restrain.

Training model using standardized data.

As the figure showed below, the loss function restrains and the accuracy
reaches a high rate.

![image](Images/5_7_Training-With-Standardization.png)

test accuracy:  0.9357080799304952


# Medium layers

(1) Get output of hidden layer in forward propagation
(2) Get output of hidden layer using hook.

_, test_fc2, output = mlpc(b_x)

test_fc2.shape:  torch.Size([64, 10])、




