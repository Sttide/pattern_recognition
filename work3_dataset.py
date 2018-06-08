# -*- coding: utf-8 -*-
# @Time    : 18-5-20 下午3:01
# @Email   : sttide@outlook.com

import tensorflow as tf
import scipy.io as sio
import numpy as np
import random
from sklearn.utils import shuffle
import os



def dataset():
    # 读入数据
    matfn = "./YaleB_32x32.mat"
    data = sio.loadmat(matfn)
    # print(data)
    # 获取特征和分类类别 30%用作train 70%用作test
    feature = np.array(data['fea'])
    gnd = np.array(data['gnd'])
    valid = int(feature.shape[0] * 0.3)
    print(feature.shape)
    print(gnd.shape)
    print(valid)  # 724
    num = np.zeros(38)
    for i in gnd:
        num[i - 1] = num[i - 1] + 1
    #print(num)
    sum = np.zeros(39)
    for i in range(len(num)):
        sum[i+1] = int(num[i] + sum[i])
    #print(sum)
    extract_num = np.zeros(38)
    for i in range(len(num)):
        extract_num[i] = int(num[i] * 0.3)
    #print(extract_num)

    # 制造标签
    new_gnd = []
    for i in range(gnd.shape[0]):
        inde = gnd[i] - 1
        f_label = np.zeros(38)
        f_label[inde] = 1
        new_gnd.append(f_label)

    train_images = []
    train_lables = []
    test_images = []
    test_lables = []


    for i in range(len(num)):
        choices = set()
        length = 0
        while length <= extract_num[i]:
            x = random.randint(0, 64)
            choices.add(x)
            length = len(choices)

        for j in range(int(num[i])):
            if j in choices:
                index = int(sum[i] + j)
                train_images.append(feature[index])
                train_lables.append(new_gnd[index])
            else:
                index =int(sum[i]+j)
                test_images.append(feature[index])
                test_lables.append(new_gnd[index])

    print(len(train_lables),len(test_images))
    return train_images,train_lables,test_images,test_lables

if __name__ == "__main__":
    dataset()
