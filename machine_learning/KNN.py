import numpy as np
from sklearn import neighbors
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

''' 训练KNN分类器 '''
clf = neighbors.KNeighborsClassifier(algorithm='kd_tree')
clf.fit(data[:500], labels[:500])

'''测试结果的打印'''
answer = clf.predict(data[500:])
print(np.mean(answer == labels[500:]))

