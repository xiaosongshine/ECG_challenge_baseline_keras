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
    return(normalize(feature).transpose())


#把标签转成oneHot
def convert2oneHot(index,Lens):
    hot = np.zeros((Lens,))
    hot[index] = 1
    return(hot)

TXT_DIR = "preliminary/reference.txt"
MANIFEST_DIR = "preliminary/reference.csv"

def create_csv(TXT_DIR=TXT_DIR):
    lists = pd.read_csv(TXT_DIR,sep=r"\t",header=None)
    lists = lists.sample(frac=1)
    lists.to_csv(MANIFEST_DIR,index=None)
    print("Finish save csv")

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
    """PRE_DIR = "sample_codes/answers.txt"
    model = load_model("best_model.15-0.88.h5")
    pre_lists = pd.read_csv(PRE_DIR,sep=r" ",header=None)
    print(pre_lists.head())
    pre_datas = np.array([get_feature(item,BASE_DIR="preliminary/TEST/") for item in pre_lists[0]])
    pre_result = model.predict_classes(pre_datas)#0-1概率预测
    print(pre_result.shape)
    pre_lists[1] = pre_result
    pre_lists.to_csv("sample_codes/answers1.txt",index=None,header=None)
    print("predict finish")"""




