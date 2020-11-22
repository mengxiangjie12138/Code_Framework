from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import random


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


# 创建100个类共10000个样本，每个样本10个特征
X, y = make_blobs(n_samples=10000, n_features=10, centers=100,random_state=0)

# 决策树
clf1 = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
scores1 = cross_val_score(clf1, data, labels)
print(scores1.mean())

# 随机森林
clf2 = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
scores2 = cross_val_score(clf2, data, labels)
print(scores2.mean())

# ExtraTree分类器集合
clf3 = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
scores3 = cross_val_score(clf3, data, labels)
print(scores3.mean())
