# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 10:59:58 2020

@author: 小小飞在路上
"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import random


class RELM_HiddenLayer:
    """
        正则化的极限学习机
        :param x: 初始化学习机时的训练集属性X
        :param num: 学习机隐层节点数
        :param C: 正则化系数的倒数
    """

    def __init__(self, x, num, C=6):
        row = x.shape[0]
        columns = x.shape[1]
        rnd = np.random.RandomState()
        # 权重w
        self.w = rnd.uniform(-1, 1, (columns, num))
        # 偏置b
        self.b = np.zeros([row, num], dtype=float)
        for i in range(num):
            rand_b = rnd.uniform(-0.4, 0.4)
            for j in range(row):
                self.b[j, i] = rand_b
        self.H0 = np.matrix(self.softplus(np.dot(x, self.w) + self.b))
        self.C = C
        self.P = (self.H0.H * self.H0 + len(x) / self.C).I
        # .T:转置矩阵,.H:共轭转置,.I:逆矩阵

    @staticmethod
    def sigmoid(x):
        """
            激活函数sigmoid
            :param x: 训练集中的X
            :return: 激活值
        """
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def softplus(x):
        """
            激活函数 softplus
            :param x: 训练集中的X
            :return: 激活值
        """
        return np.log(1 + np.exp(x))

    @staticmethod
    def tanh(x):
        """
            激活函数tanh
            :param x: 训练集中的X
            :return: 激活值
        """
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    # 分类问题 训练
    def classifisor_train(self, T):
        """
            初始化了学习机后需要传入对应标签T
            :param T: 对应属性X的标签T
            :return: 隐层输出权值beta
        """
        if len(T.shape) > 1:
            pass
        else:
            self.en_one = OneHotEncoder()
            T = self.en_one.fit_transform(T.reshape(-1, 1)).toarray()
            pass
        all_m = np.dot(self.P, self.H0.H)
        self.beta = np.dot(all_m, T)
        return self.beta

    # 分类问题 测试
    def classifisor_test(self, test_x):
        """
            传入待预测的属性X并进行预测获得预测值
            :param test_x:被预测标签的属性X
            :return: 被预测标签的预测值T
        """
        b_row = test_x.shape[0]
        h = self.softplus(np.dot(test_x, self.w) + self.b[:b_row, :])
        result = np.dot(h, self.beta)
        result = np.argmax(result, axis=1)
        return result


# url = 'C:/Users/weixifei/Desktop/TensorFlow程序/data1.csv'
# data = pd.read_csv(url, sep=',', header=None)
# data = np.array(data)
# data = shuffle(data)
# X_data = data[:, :23]
# Y = data[:, 23]
# labels = np.asarray(pd.get_dummies(Y), dtype=np.int8)
#
# num_train = 0.3
# X_train, X_, Y_train, Y_ = train_test_split(X_data, labels, test_size=num_train, random_state=20)
# X_test, X_vld, Y_test, Y_vld = train_test_split(X_, Y_, test_size=0.1, random_state=20)
#
# stdsc = StandardScaler()
# X_train = stdsc.fit_transform(X_train)
# X_test = stdsc.fit_transform(X_test)
# X_vld = stdsc.fit_transform(X_vld)
# Y_true = np.argmax(Y_test, axis=1)

def get_data(data_path, labels_path):
    data = np.load(data_path)
    labels = np.load(labels_path)
    data_shuffle_list = []
    for i in range(len(labels)):
        data_shuffle_list.append((data[i], labels[i]))
    random.shuffle(data_shuffle_list)
    data_new = []
    labels_new = []
    for i in range(len(labels)):
        data_new.append(data_shuffle_list[i][0])
        labels_new.append(data_shuffle_list[i][1])
    return np.array(data_new), np.array(labels_new)


data_path = r'D:\My_Code\mengxiangjie\DTGCN\evaluation2\resnet18_features_sixclassify\features.npy'
labels_path = r'D:\My_Code\mengxiangjie\DTGCN\evaluation2\resnet18_features_sixclassify\label_true.npy'
data, labels = get_data(data_path, labels_path)

for j in range(1, 300, 50):
    a = RELM_HiddenLayer(data[:500], j)
    a.classifisor_train(labels[:500])
    predict = a.classifisor_test(data[500:])
    acc = metrics.precision_score(predict, labels[500:], average='macro')
    # acc = metrics.recall_score(predict, labels[500:], average='macro')
    print('hidden- %d,acc：%f' % (j, acc))




