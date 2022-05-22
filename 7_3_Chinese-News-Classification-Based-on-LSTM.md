# 7.3 Chinese News Classification Based on LSTM


# 1. Code 
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torchtext.legacy.data

from matplotlib.font_manager import FontProperties
fonts = FontProperties(fname='/Library/Fonts/华文细黑.ttf')
import re
import string
import copy
import time

from sklearn.metrics import  accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import jieba
from torchtext.legacy import data
from torchtext.vocab import Vectors


def visualHighestFreq(TEXT):
    word_freq = TEXT.vocab.freqs.most_common(n=50)
    word_freq = pd.DataFrame(data=word_freq,columns=['word','frequency'])
    word_freq.plot(x='word',y='frequency',kind='bar',legend=False,figsize=(12,7))
    plt.xticks(rotation=90,fontproperties=fonts,size=10)
    plt.title('Figure 7.7 Words with High Frequency')
    plt.show()

train_df = pd.read_csv('data/Text/cnews/cnews_train.txt',sep='\t',
                       header=None,names=['label','text'])
val_df = pd.read_csv('data/Text/cnews/cnews_val.txt',sep='\t',
                     header=None,names=['label','text'])
test_df = pd.read_csv('data/Text/cnews/cnews_test.txt',sep='\t',
                      header=None,names=['label','text'])
stop_words = pd.read_csv('data/Text/cnews/cnews_vocab.txt',
                         engine='python',
                         header=None,names=['text'],
                         error_bad_lines=False
                         )

def visualTrainingProcess(train_process):
    plt.figure(figsize=(18,6))
    plt.subplot(1,2,1)
    plt.plot(train_process.epoch,train_process.train_loss_all,'r-',label='Train Loss')
    plt.plot(train_process.epoch, train_process.val_loss_all, 'bs-', label = 'Val loss')
    plt.legend()
    plt.xlabel('Epoch number: ',size=14)
    plt.ylabel('Loss value: ',size=14)

    plt.subplot(1,2,2)
    plt.plot(train_process.epoch, train_process.train_acc_all, 'r-', label='Train Acc')
    plt.plot(train_process.epoch, train_process.val_acc_all,'bs-',label='Val Acc')
    plt.xlabel('Epoch number: ',size=14)
    plt.ylabel('Acc: ',size=14)
    plt.legend()
    plt.show()

def visualConfMatrix(test_y_all, pre_lab_all):
    class_label = ['体育', '娱乐', '家居', '房产', '教育',
                   '时尚', '时政', '游戏', '科技', '财经']

    conf_mat = confusion_matrix(test_y_all, pre_lab_all)
    df_cm = pd.DataFrame(conf_mat, index=class_label, columns=class_label)
    heatmap = sns.heatmap(df_cm, annot=True, fmt='d', cmap='YlGnBu')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(),
                                 rotation=0, ha='right', fontproperties=fonts)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(),
                                 rotation=45, ha='right', fontproperties=fonts)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Figure 7-9 Confusion Matrix Heatmap on Test Set')
    plt.show()
    plt.savefig('./Images/7_9-Confusion-Matrix-Heatmap-on-Test-Set.png')

def visualWordVecDistribution():
    from sklearn.manifold import TSNE
    # import the saved model
    lstmmodel = torch.load('data/model/lstmmodel.pkl')
    # get word vectors
    word2vec = lstmmodel.embedding.weight
    # corresponding words  of word vectors
    words = TEXT.vocab.itos
    # use tsne to squeeze the word vector and visualize the distribution of all words
    tsne = TSNE(n_components=2, random_state=123)
    word2vec_tsne = tsne.fit_transform(word2vec.data.numpy())
    # use scatter to visualize the distribution of all words
    plt.figure(figsize=(10,8))
    plt.scatter(word2vec_tsne[:,0], word2vec_tsne[:,1], s=4)
    plt.title('Distribution of All Words',fontproperties= fonts, size=15)
    plt.show()

# def visualHighFreqWords():



def chinese_pre(text_data):
    # convert letters to lower cases and remove numbers
    text_data = text_data.lower()
    text_data = re.sub('\d+','',text_data)
    # word split using precise mode
    text_data = list(jieba.cut(text_data,cut_all=False))
    # remove stop words and additional spaces
    text_data = [word.strip() for word in text_data if word not in stop_words.text.values]
    # concat processed words into strings by using spaces
    text_data = ' '.join(text_data)
    return text_data

# word split
train_df['cutword'] = train_df.text.apply(chinese_pre)
val_df['cutword'] = val_df.text.apply(chinese_pre)
test_df['cutword'] = test_df.text.apply(chinese_pre)
train_df.cutword.head()

# recode labels of the text
labelMap = {'体育': 0, '娱乐': 1, '家居': 2, '房产': 3, '教育': 4,
            '时尚': 5, '时政': 6, '游戏': 7, '科技': 8, '财经': 9}
train_df['labelcode'] = train_df['label'].map(labelMap)
val_df['labelcode'] = val_df['label'].map(labelMap)
test_df['labelcode'] = test_df['label'].map(labelMap)

# save preprocessed text cutword and recoded labels labelcode
train_df[['labelcode','cutword']].to_csv('data/Text/cnews/cnews_train2.csv',index=False)
val_df[['labelcode','cutword']].to_csv('data/Text/cnews/cnews_val2.csv',index=False)
test_df[['labelcode','cutword']].to_csv('data/Text/cnews/cnews_test2.csv',index=False)


# data preparation using torchtext
mytokenize = lambda x: x.split()
TEXT = data.Field(sequential=True,tokenize=mytokenize,
                  include_lengths=True,use_vocab=True,
                  batch_first=True,fix_length=400)
LABEL = data.Field(sequential=False, use_vocab=False,
                   pad_token=None, unk_token=None)
# process data by columns
text_data_fields = [
    ('labelcode',LABEL),
    ('cutword', TEXT)
]

# data loading
traindata,valdata, testdata = data.TabularDataset.splits(
    path='data/Text/cnews',format='csv',
    train='cnews_train2.csv', fields=text_data_fields,
    validation='cnews_val2.csv',
    test = 'cnews_test2.csv',skip_header=True
)

print('Length of traindata: ',len(traindata))
print('Length of valdata: ', len(valdata))
print('Length of testdata: ',len(testdata))

# construct vocabulary using train set without pretrained word vector
TEXT.build_vocab(traindata, maxsize=20000,vectors=None)
LABEL.build_vocab(traindata)

# visualize top 50 highest frequent words
# visualHighestFreq(TEXT)

# dataloader
BATCH_SIZE = 64
train_iter = data.BucketIterator(traindata, batch_size=BATCH_SIZE)
val_iter = data.BucketIterator(valdata,batch_size=BATCH_SIZE)
test_iter = data.BucketIterator(testdata,batch_size=BATCH_SIZE)



# construct LSTM network
class LSTMNet(nn.Module):
    def __init__(self,vocab_size, embedding_dim,hidden_dim,layer_dim,output_dim):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        # word vector embedding
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        # LSTM + fully connected layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layer_dim, batch_first=True)
        self.fcl = nn.Linear(hidden_dim, output_dim)
    def forward(self,x):
        embeds = self.embedding(x)
        r_out, (h_n, hc) = self.lstm(embeds,None)
        out = self.fcl(r_out[:, -1, :])
        return out

vocab_size = len(TEXT.vocab)
embedding_dim = 100
hidden_dim = 128
layer_dim = 1
output_dim = 10
lstmmodel = LSTMNet(vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim)

# print(lstmmodel)

def train_model2(model, traindataloader, valdataloader, criterion,
                 optimizer, num_epochs = 25,):
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    since = time.time()
    for epoch in range(num_epochs):
        print('-'*10)
        print('Epoch {}/{}'.format(epoch,num_epochs))
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0.0
        val_corrects = 0
        val_num = 0

        model.train()
        for step,batch in enumerate(traindataloader):
            textdata, target = batch.cutword[0], batch.labelcode,
            out = model(textdata)
            pre_lab = torch.argmax(out,1)
            loss = criterion(out,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(target)
            train_corrects += torch.sum(pre_lab==target.data)
            train_num = len(target)
        train_loss_all.append(train_loss/train_num)
        train_acc_all.append(train_corrects.double().item()/train_num)
        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(
            epoch, train_loss_all[-1],train_acc_all[-1]))

        model.eval()
        for step, batch in enumerate(valdataloader):
            textdata, target = batch.cutword[0], batch.labelcode.view(-1)
            out = model(textdata)
            pre_lab = torch.argmax(out,1)
            loss = criterion(out,target)
            val_loss += loss.item() * len(target)
            val_corrects += torch.sum(pre_lab== target.data)
            val_num += len(target)
        # compute loss and accuracy on test set during 1 epoch
        val_loss_all.append(val_loss/val_num)
        val_acc_all.append(val_corrects.double().item()/val_num)
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(
            epoch, val_loss_all[-1], val_acc_all[-1]
        ))
    train_process = pd.DataFrame(
        data={'epoch': range(num_epochs),
            'train_loss_all': train_loss_all,
            'train_acc_all': train_acc_all,
            'val_loss_all': val_loss_all,
            'val_acc_all': val_acc_all})
    return model, train_process


# optimizer = torch.optim.Adam(lstmmodel.parameters(),lr=0.0003)
# loss_func = nn.CrossEntropyLoss()
# lstmmodel, train_process = train_model2(lstmmodel,train_iter,val_iter,loss_func,
#                                         optimizer,num_epochs=20)

# visualize the training process
# visualTrainingProcess(train_process)


lstmmodel.eval()
test_y_all = torch.LongTensor()
pre_lab_all = torch.LongTensor()

for step, batch in enumerate(test_iter):
    textdata, target = batch.cutword[0],batch.labelcode.view(-1)
    out = lstmmodel(textdata)
    pre_lab = torch.argmax(out,1)

acc = accuracy_score(test_y_all, pre_lab_all)
print('Accuracy on test set is: ',acc)
# compute confusion matrix and visualize
# visualConfMatrix(test_y_all,pre_lab_all)


# visualize distribution of word vectors
# visualWordVecDistribution()
```

# 2. Illustration

## 2.1 Data Preparation

Here we use **cnews** dataset, which is a subset of THUCNews.

Training set: 5000 * 10

Validation set: 500 * 10

Test set: 1000 * 10

## 2.2 Data Loading and Preprocessing

Here we use ```chinese_pre()``` to preprocess the text.

## 2.3 Traing of LSTM

## 2.4 Prediction of LSTM

## 2.5 Visualization

Here we visualize the distribution of word vectors.



We can also visualize some words with high frequency.



