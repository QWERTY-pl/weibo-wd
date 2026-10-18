# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:36:12 2020

@author: Lukas
"""

#加载必要的模块
import pandas as pd
# 读取文本数据

data = pd.read_csv('weibo_senti_100k.csv')
data = data.dropna()   #去掉数据集的空值
data.shape  #输出数据结构
data.head()  # 输出文本数据集的前5行

# ## 结巴分词 

import jieba
data['data_cut'] = data['review'].apply(lambda x: list(jieba.cut(x)))  #内嵌自定义函数来分词
data.head()


# ##  去停用词

# 读取停用词
with open('stopword.txt','r',encoding = 'utf-8') as f:  #读取停用词
    stop = f.readlines()
# 对停用词处理
import re
stop = [re.sub(' |\n|\ufeff','',r) for r in stop]   #替换停用词表的空格等

# 去除停用词
#把分词之后的文本根据停用词表去掉停用词
data['data_after'] = [[i for i in s if i not in stop] for s in data['data_cut']]
data.head()


# 构建词云
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# ## 词频统计
# 重组词组
num_words = [''.join(i) for i in data['data_after']] #把所有词组提取出来
num_words = ''.join(num_words)  #词组放在num_words上
num_words= re.sub(' ','',num_words)

# 计算全部词频
num = pd.Series(jieba.lcut(num_words)).value_counts()
# 用wordcloud画图
wc_pic = WordCloud(background_color='white',font_path=r'C:\Windows\Fonts\simhei.ttf').fit_words(num)
plt.figure(figsize=(10,10))  #图片大小定义
plt.imshow(wc_pic)#输出图片
plt.axis('off')#不显示坐标轴
plt.show()



# 构建词向量矩阵
w = [] 
for i in data['data_after']:  
    w.extend(i)  #将所有词语整合在一起  
num_data = pd.DataFrame(pd.Series(w).value_counts()) # 计算出所有单词的个数

num_data['id'] = list(range(1,len(num_data)+1))   #把这些数据增加序号

#  转化成数字
a = lambda x:list(num_data['id'][x])    #以序号为序定义实现函数
data['vec'] = data['data_after'].apply(a)  #apply（）方法实现
data.head()

# ##  数据集划分
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence  

maxlen = 128   #句子长度
vec_data = list(sequence.pad_sequences(data['vec'],maxlen=maxlen))   #把文本数据都统一长度
x,xt,y,yt = train_test_split(vec_data,data['label'],test_size = 0.2,random_state = 123)   #分割训练集--2-8原则

# 转换数据类型
import numpy as np
x = np.array(list(x))
y = np.array(list(y))
xt = np.array(list(xt))
yt = np.array(list(yt))

###由于数据量较大，运行时间太长，可选择其中的2000条训练，500条测试来验证模型
x=x[:2000,:]
y=y[:2000]
xt=xt[:500,:]
yt=yt[:500]


# SVM方法
from sklearn.svm import SVC
 
clf = SVC(C=1, kernel = 'linear')  # 用自带的 'rbf'
clf.fit(x,y)   #  #模型训练

# 调用报告
from sklearn.metrics import classification_report
test_pre = clf.predict(xt)  # 模型预测
report = classification_report(yt,test_pre)   #预测结果
print(report)

from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding  
from keras.layers.recurrent import LSTM, GRU



model = Sequential()  
model.add(Embedding(len(num_data['id'])+1,256))   # 输入层，词向量表示层
model.add(Dense(32, activation='sigmoid', input_dim=100))  # 全连接层，32层
model.add(LSTM(128))   # LSTM网络层
model.add(Dense(1))  # 全连接层--输出层
model.add(Activation('sigmoid'))    # 输出层的激活函数
model.summary()  #输出模型结构


#模型的画图表示
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from keras.utils import plot_model
plot_model(model,to_file='Lstm2.png',show_shapes=True)
ls = mpimg.imread('Lstm2.png') # 读取和代码处于同一目录下的Lstm.png
plt.imshow(ls) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()


model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=["accuracy"])  #模型融合

#训练模型
model.fit(x,y,validation_data=(x,y),epochs=15)


#模型验证
loss,accuracy=model.evaluate(xt,yt,batch_size=12)  # 测试集评估
print('Test loss:',loss)
print('Test accuracy:', accuracy)





