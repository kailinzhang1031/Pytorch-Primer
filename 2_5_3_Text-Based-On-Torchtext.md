# 2.5.3 Text Based On Torchtext


Here we introduce several samples in practical NLP tasks based on **torchtext**.

Relevant learning: 

[NLP preliminary task from stretch.]()

## 1. Code

```python
from  torchtext.legacy import data

mytokenize = lambda x:x.split()


text = data.Field(sequential=True,
                  tokenize=mytokenize,
                  use_vocab=True,
                  batch_first=True,
                  fix_length=200)


label = data.Field(sequential=False,
                   use_vocab=False,
                   pad_token=None,
                   unk_token=None)


text_data_fields = [
    ('label',label),
    ('text',text)
]

traindata, testdata = data.TabularDataset.splits(
    path='data/chap2/textdata/',
    format = 'csv',
    train='train.csv',fields=text_data_fields,
    test='test.csv',skip_header=True
)

print(len(traindata))
print(len(testdata))


text.build_vocab(traindata,max_size=1000,vectors=None)

train_iter = data.BucketIterator(traindata,batch_size=4)
test_iter = data.BucketIterator(testdata,batch_size=4)

# for step, batch in enumerate(train_iter):
#     print(batch.label)
#     print(batch.label.shape)
#     print(batch.text)
#     print(batch.text.shape)
#     print('--------------------')

```

## 2. Illustration

- train.csv
    ```python
    label,text
    1,asian exporters fear damage from u . s .- japan rift mounting trade friction between the u . s . and japan has raised fears among many of asia ' s exporting nations that the row could inflict far - reaching economic damage , businessmen and officials said .
    0,they told reuter correspondents in asian capitals a u . s . move against japan might boost protectionist sentiment in the u . s . and lead to curbs on american imports of their products .
    1,but some exporters said that while the conflict would hurt them in the long - run , in the short - term tokyo ' s loss might be their gain .
    0,the u . s . has said it will impose 300 mln dlrs of tariffs on imports of japanese electronics goods on april 17 , in retaliation for japan ' s alleged failure to stick to a pact not to sell semiconductors on world markets at below cost .
    ```

- test.csv

    ```python
    label,text
    1,unofficial japanese estimates put the impact of the tariffs at 10 billion dlrs and spokesmen for major electronics firms said they would virtually halt exports of products hit by the new taxes .
    0," we wouldn ' t be able to do business ," said a spokesman for leading japanese electronics firm matsushita electric industrial co ltd & lt ; mc . t >.
    1," if the tariffs remain in place for any length of time beyond a few months it will mean the complete erosion of exports ( of goods subject to tariffs ) to the u . s .," said tom murtha , a stock analyst at the tokyo office of broker & lt ; james capel and co >.
    0,in taiwan , businessmen and officials are also worried .
    ```

## 2. Dataset Construction

### 2.1 Defining tokenize method.
```python
from  torchtext.legacy import data
    
mytokenize = lambda x:x.split()
```

### 2.2 Constructing structural data.

```python
text = data.Field(sequential=True,
                    tokenize=mytokenize,
                    use_vocab=True,
                    batch_first=True,
                    fix_length=200)
    
label = data.Field(sequential=False,
                    use_vocab=False,
                    pad_token=None,
                    unk_token=None)
    
text_data_fields = [
    ('label',label),
    ('text',text)
]
    
traindata,testdata = data.TabularDataset.splits(
path='data/chap2/textdata',
format = 'csv',
train='train.csv',fields=text_data_fields,
test='test.csv',skip_header=True
)
```


### 2.3 Vocabulary construction
```python
text.build_vocab(traindata,max_size=1000,vectors=None)
```

A vocabulary is created in the type of **torchtext.legacy.vocab.Vocab**.

This vocabulary is subscriptable:

text.vocab['asia'] = 28

#### 2.4 Dataloader Instantiation

```python
train_iter = data.BucketIterator(traindata,batch_size=4)
test_iter = data.BucketIterator(testdata,batch_size=4)
```

## 3. Check

- ```batch.label```: torch.tensor of torch.Size([4])
    ```python
    tensor([0, 0, 1, 1])
    ```
- ```batch.text```: torch.tensor of torch.Size([4, 200])
    ```python
    tensor([[73, 75, 64, 34,  7, 12, 32, 23,  5,  2,  3,  2, 58, 24,  8, 55, 30, 61,
             68,  7,  4,  5,  2,  3,  2, 11, 52, 74, 36,  9, 25, 16,  6, 71, 60,  2,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1],
            [ 4,  5,  2,  3,  2, 15, 17, 50, 78, 48, 22, 56, 37,  6, 70,  9, 16,  6,
             51, 39, 46,  9, 27, 21,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1],
            [12, 14, 42, 13, 45,  5,  2,  3, 20,  8, 65, 57, 76, 44, 29,  4,  5,  2,
              3,  2, 11,  8, 15, 62, 43, 26, 54,  6, 28, 19,  3, 40, 59, 18,  4, 66,
             35, 49, 41, 10, 63, 38, 13,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1],
            [31, 69, 14, 17, 18, 77,  4, 33, 79, 47, 72,  7,  4, 53, 10, 67,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1]])
    ```










