
## [深度应用]·基于KerasConv1D心电图检测实战开源教程

> 个人主页-->  [http://www.yansongsong.cn/](https://link.zhihu.com/?target=http%3A//www.yansongsong.cn/)  
> 项目github地址：[https://github.com/xiaosongshine/preliminary_challenge_baseline_keras](https://link.zhihu.com/?target=https%3A//github.com/xiaosongshine/preliminary_challenge_baseline_keras)

## 实战概述

本实战内容取自笔者参加的首届中国心电智能大赛项目，初赛要求为设计一个自动识别心电图波形算法。笔者使用Keras框架设计了基于Conv1D结构的模型，并且开源了代码作为Baseline。内容包括数据预处理，模型搭建，网络训练，模型应用等，此Baseline采用最简单的一维卷积达到了88%测试准确率。有多支队伍在笔者基线代码基础上调优取得了优异成绩，顺利进入复赛。

  

## 大赛简介

为响应国家健康中国战略，推送健康医疗和大数据的融合发展的政策，由清华大学临床医学院和数据科学研究院，天津市武清区京津高村科技创新园，以及多家重点医院联合主办的首届中国心电智能大赛正式启动。自今日起至2019年3月31日24时，大赛开启全球招募，预计大赛总奖金将高达百万元！目前官方报名网站已上线，欢迎高校、医院、创业团队等有志于中国心电人工智能发展的人员踊跃参加。

首届中国心电智能大赛官方报名网站>>[http://mdi.ids.tsinghua.edu.cn](https://link.zhihu.com/?target=http%3A//mdi.ids.tsinghua.edu.cn/)

  

## 数据介绍

下载完整的训练集和测试集，共1000例常规心电图，其中训练集中包含600例，测试集中共400例。该数据是从多个公开数据集中获取。参赛团队需要利用有正常/异常两类标签的训练集数据设计和实现算法，并在没有标签的测试集上做出预测。

该心电数据的采样率为500 Hz。为了方便参赛团队用不同编程语言都能读取数据，所有心电数据的存储格式为MAT格式。该文件中存储了12个导联的电压信号。训练数据对应的标签存储在txt文件中，其中0代表正常，1代表异常。

  

## 赛题分析

简单分析一下，初赛的数据集共有1000个样本，其中训练集中包含600例，测试集中共400例。其中训练集中包含600例是具有label的，可以用于我们训练模型；测试集中共400例没有标签，需要我们使用训练好的模型进行预测。

赛题就是一个二分类预测问题，解题思路应该包括以下内容

1.  数据读取与处理
2.  网络模型搭建
3.  模型的训练
4.  模型应用与提交预测结果

  

## 实战应用

经过对赛题的分析，我们把任务分成四个小任务，首先第一步是：

### 1.数据读取与处理

该心电数据的采样率为500 Hz。为了方便参赛团队用不同编程语言都能读取数据，所有心电数据的存储格式为MAT格式。该文件中存储了12个导联的电压信号。训练数据对应的标签存储在txt文件中，其中0代表正常，1代表异常。

我们由上述描述可以得知，

-   我们的数据保存在MAT格式文件中（**这决定了后面我们要如何读取数据**）
-   采样率为500 Hz（这个信息并没有怎么用到，大家可以简单了解一下，就是1秒采集500个点，由后面我们得知每个数据都是5000个点，也就是10秒的心电图片）
-   12个导联的电压信号（这个是指采用12种导联方式，大家可以简单理解为用12个体温计量体温，从而得到更加准确的信息，下图为导联方式简单介绍，大家了解下即可。**要注意的是，既然提供了12种导联，我们应该全部都用到，虽然我们仅使用一种导联方式也可以进行训练与预测，但是经验告诉我们，采取多个特征会取得更优效果**）

  

  

![](https://pic4.zhimg.com/80/v2-f657755203861f52f5b03023900a2087_hd.jpg)

​

  

**数据处理函数定义：**

```python
import keras
from scipy.io import loadmat
import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd
import math
import os
from keras.layers import *
from keras.models import *
from keras.objectives import *


BASE_DIR = "preliminary/TRAIN/"

#进行归一化
def normalize(v):
    return (v - v.mean(axis=1).reshape((v.shape[0],1))) / (v.max(axis=1).reshape((v.shape[0],1)) + 2e-12)

#loadmat打开文件
def get_feature(wav_file,Lens = 12,BASE_DIR=BASE_DIR):
    mat = loadmat(BASE_DIR+wav_file)
    dat = mat["data"]
    feature = dat[0:12]
    return(normalize(feature).transopse())


#把标签转成oneHot形式
def convert2oneHot(index,Lens):
    hot = np.zeros((Lens,))
    hot[index] = 1
    return(hot)

TXT_DIR = "preliminary/reference.txt"
MANIFEST_DIR = "preliminary/reference.csv"
```

  

**读取一条数据进行显示**

```python
if __name__ == "__main__":
    dat1 = get_feature("preliminary/TRAIN/TRAIN101.mat")
    print(dat1.shape)
    #one data shape is (12, 5000)
    plt.plt(dat1[:,0])
    plt.show()
```

  

  

![](https://pic4.zhimg.com/80/v2-b218248bc5935f2b7f77afa575fb8a63_hd.jpg)

​

我们由上述信息可以看出每种导联都是由5000个点组成的列表，12种导联方式使每个样本都是12*5000的矩阵，类似于一张分辨率为12x5000的照片。

我们需要处理的就是把每个读取出来，归一化一下，送入网络进行训练可以了。

**标签处理方式**

```python
def create_csv(TXT_DIR=TXT_DIR):
    lists = pd.read_csv(TXT_DIR,sep=r"\t",header=None)
    lists = lists.sample(frac=1)
    lists.to_csv(MANIFEST_DIR,index=None)
    print("Finish save csv")
```

  

我这里是采用从reference.txt读取，然后打乱保存到reference.csv中，**注意一定要进行数据打乱操作，不然训练效果很差。因为原始数据前面便签全部是1，后面全部是0**

**数据迭代方式**

```python
Batch_size = 20
def xs_gen(path=MANIFEST_DIR,batch_size = Batch_size,train=True):

    img_list = pd.read_csv(path)
    if train :
        img_list = np.array(img_list)[:500]
        print("Found %s train items."%len(img_list))
        print("list 1 is",img_list[0])
        steps = math.ceil(len(img_list) / batch_size)    # 确定每轮有多少个batch
    else:
        img_list = np.array(img_list)[500:]
        print("Found %s test items."%len(img_list))
        print("list 1 is",img_list[0])
        steps = math.ceil(len(img_list) / batch_size)    # 确定每轮有多少个batch
    while True:
        for i in range(steps):

            batch_list = img_list[i * batch_size : i * batch_size + batch_size]
            np.random.shuffle(batch_list)
            batch_x = np.array([get_feature(file) for file in batch_list[:,0]])
            batch_y = np.array([convert2oneHot(label,2) for label in batch_list[:,1]])

            yield batch_x, batch_y
```

  

数据读取的方式我采用的是生成器的方式，这样可以按batch读取，加快训练速度，大家也可以采用一下全部读取，看个人的习惯了

  

### 2.网络模型搭建

数据我们处理好了，后面就是模型的搭建了，我使用keras搭建的，操作简单便捷，tf，pytorch，sklearn大家可以按照自己喜好来。

网络模型可以选择CNN，RNN，Attention结构，或者多模型的融合，抛砖引玉，此Baseline采用的一维CNN方式，[一维CNN学习地址](https://link.zhihu.com/?target=https%3A//blog.csdn.net/xiaosongshine/article/details/88614450)

**模型搭建**

```python
TIME_PERIODS = 5000
num_sensors = 12
def build_model(input_shape=(TIME_PERIODS,num_sensors),num_classes=2):
    model = Sequential()
    #model.add(Reshape((TIME_PERIODS, num_sensors), input_shape=input_shape))
    model.add(Conv1D(16, 16,strides=2, activation='relu',input_shape=input_shape))
    model.add(Conv1D(16, 16,strides=2, activation='relu',padding="same"))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 8,strides=2, activation='relu',padding="same"))
    model.add(Conv1D(64, 8,strides=2, activation='relu',padding="same"))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(128, 4,strides=2, activation='relu',padding="same"))
    model.add(Conv1D(128, 4,strides=2, activation='relu',padding="same"))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(256, 2,strides=1, activation='relu',padding="same"))
    model.add(Conv1D(256, 2,strides=1, activation='relu',padding="same"))
    model.add(MaxPooling1D(2))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    return(model)
```

  

**用model.summary()输出的网络模型为**

```bash
________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
reshape_1 (Reshape)          (None, 5000, 12)          0
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 2493, 16)          3088
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 1247, 16)          4112
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 623, 16)           0
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 312, 64)           8256
_________________________________________________________________
conv1d_4 (Conv1D)            (None, 156, 64)           32832
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, 78, 64)            0
_________________________________________________________________
conv1d_5 (Conv1D)            (None, 39, 128)           32896
_________________________________________________________________
conv1d_6 (Conv1D)            (None, 20, 128)           65664
_________________________________________________________________
max_pooling1d_3 (MaxPooling1 (None, 10, 128)           0
_________________________________________________________________
conv1d_7 (Conv1D)            (None, 10, 256)           65792
_________________________________________________________________
conv1d_8 (Conv1D)            (None, 10, 256)           131328
_________________________________________________________________
max_pooling1d_4 (MaxPooling1 (None, 5, 256)            0
_________________________________________________________________
global_average_pooling1d_1 ( (None, 256)               0
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 514
=================================================================
Total params: 344,482
Trainable params: 344,482
Non-trainable params: 0
_________________________________________________________________
```

  

训练参数比较少，大家可以根据自己想法更改。

### 3.网络模型训练

**模型训练**

```python
if __name__ == "__main__":
    """dat1 = get_feature("TRAIN101.mat")
    print("one data shape is",dat1.shape)
    #one data shape is (12, 5000)
    plt.plot(dat1[0])
    plt.show()"""
    if (os.path.exists(MANIFEST_DIR)==False):
        create_csv()
    train_iter = xs_gen(train=True)
    test_iter = xs_gen(train=False)
    model = build_model()
    print(model.summary())
    ckpt = keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_acc:.2f}.h5',
        monitor='val_acc', save_best_only=True,verbose=1)
    model.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])
    model.fit_generator(
        generator=train_iter,
        steps_per_epoch=500//Batch_size,
        epochs=20,
        initial_epoch=0,
        validation_data = test_iter,
        nb_val_samples = 100//Batch_size,
        callbacks=[ckpt],
        )
```

  

**训练过程输出（最优结果：loss: 0.0565 - acc: 0.9820 - val_loss: 0.8307 - val_acc: 0.8800）**

```bash
Epoch 10/20
25/25 [==============================] - 1s 37ms/step - loss: 0.2329 - acc: 0.9040 - val_loss: 0.4041 - val_acc: 0.8700

Epoch 00010: val_acc improved from 0.85000 to 0.87000, saving model to best_model.10-0.87.h5
Epoch 11/20
25/25 [==============================] - 1s 38ms/step - loss: 0.1633 - acc: 0.9380 - val_loss: 0.5277 - val_acc: 0.8300

Epoch 00011: val_acc did not improve from 0.87000
Epoch 12/20
25/25 [==============================] - 1s 40ms/step - loss: 0.1394 - acc: 0.9500 - val_loss: 0.4916 - val_acc: 0.7400

Epoch 00012: val_acc did not improve from 0.87000
Epoch 13/20
25/25 [==============================] - 1s 38ms/step - loss: 0.1746 - acc: 0.9220 - val_loss: 0.5208 - val_acc: 0.8100

Epoch 00013: val_acc did not improve from 0.87000
Epoch 14/20
25/25 [==============================] - 1s 38ms/step - loss: 0.1009 - acc: 0.9720 - val_loss: 0.5513 - val_acc: 0.8000

Epoch 00014: val_acc did not improve from 0.87000
Epoch 15/20
25/25 [==============================] - 1s 38ms/step - loss: 0.0565 - acc: 0.9820 - val_loss: 0.8307 - val_acc: 0.8800

Epoch 00015: val_acc improved from 0.87000 to 0.88000, saving model to best_model.15-0.88.h5
Epoch 16/20
25/25 [==============================] - 1s 38ms/step - loss: 0.0261 - acc: 0.9920 - val_loss: 0.6443 - val_acc: 0.8400

Epoch 00016: val_acc did not improve from 0.88000
Epoch 17/20
25/25 [==============================] - 1s 38ms/step - loss: 0.0178 - acc: 0.9960 - val_loss: 0.7773 - val_acc: 0.8700

Epoch 00017: val_acc did not improve from 0.88000
Epoch 18/20
25/25 [==============================] - 1s 38ms/step - loss: 0.0082 - acc: 0.9980 - val_loss: 0.8875 - val_acc: 0.8600

Epoch 00018: val_acc did not improve from 0.88000
Epoch 19/20
25/25 [==============================] - 1s 37ms/step - loss: 0.0045 - acc: 1.0000 - val_loss: 1.0057 - val_acc: 0.8600

Epoch 00019: val_acc did not improve from 0.88000
Epoch 20/20
25/25 [==============================] - 1s 37ms/step - loss: 0.0012 - acc: 1.0000 - val_loss: 1.1088 - val_acc: 0.8600

Epoch 00020: val_acc did not improve from 0.88000
```

### 4.模型应用预测结果

**预测数据**

```python
if __name__ == "__main__":
    
    PRE_DIR = "sample_codes/answers.txt"
    model = load_model("best_model.15-0.88.h5")
    pre_lists = pd.read_csv(PRE_DIR,sep=r" ",header=None)
    print(pre_lists.head())
    pre_datas = np.array([get_feature(item,BASE_DIR="preliminary/TEST/") for item in pre_lists[0]])
    pre_result = model.predict_classes(pre_datas)#0-1概率预测
    print(pre_result.shape)
    pre_lists[1] = pre_result
    pre_lists.to_csv("sample_codes/answers1.txt",index=None,header=None)
    print("predict finish")
```

  

下面是前十条预测结果：

```bash
TEST394,0
TEST313,1
TEST484,0
TEST288,0
TEST261,1
TEST310,0
TEST286,1
TEST367,1
TEST149,1
TEST160,1
```

  

  

## **展望**

此Baseline采用最简单的一维卷积达到了88%测试准确率（可能会因为随机初始化值上下波动），大家也可以多尝试GRU，Attention，和Resnet等结果，测试准确率准确率会突破90+。

能力有限，写的不好的地方欢迎大家批评指正。。

> 个人主页-->  [http://www.yansongsong.cn/](https://link.zhihu.com/?target=http%3A//www.yansongsong.cn/)  
> 项目github地址：[https://github.com/xiaosongshine/preliminary_challenge_baseline_keras](https://link.zhihu.com/?target=https%3A//github.com/xiaosongshine/preliminary_challenge_baseline_keras)

**欢迎Fork+Star，觉得有用的话，麻烦小小鼓励一下 ><**
