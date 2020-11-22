import numpy as np
import kmeans_feature_extractor.config as kmeans_features_config


def test_labels(l):
    l_raw = np.load('label_true.npy')
    class1 = kmeans_features_config.class1_num
    class2 = kmeans_features_config.class2_num + class1
    class3 = kmeans_features_config.class3_num + class2
    l1 = []
    for i in range(len(l_raw)):
        l1.append(1)
    for index, i in enumerate(l):
        if i <= class1:
            l1[index] = l_raw[0]
        elif class1 < i <= class2:
            l1[index] = l_raw[class1 + 1]
        elif class2 < i <= class3:
            l1[index] = l_raw[class2 + 1]
    return l1


def test_acc(labels_):
    labels_true = np.load('label_true.npy')
    label_pred = labels_

    i1_raw = []
    i2_raw = []
    i3_raw = []
    for i, label in enumerate(label_pred):
        if label == 0:
            i1_raw.append(i)
        elif label == 1:
            i2_raw.append(i)
        elif label == 2:
            i3_raw.append(i)
    i1 = test_labels(i1_raw)
    i2 = test_labels(i2_raw)
    i3 = test_labels(i3_raw)
    label1_list = sorted(i1, key=i1.count, reverse=True)
    label2_list = sorted(i2, key=i2.count, reverse=True)
    label3_list = sorted(i3, key=i3.count, reverse=True)
    label1 = label1_list[0]
    if label2_list[0] == label1:
        label2 = list(set(label2_list))[1]
    else:
        label2 = label2_list[0]
    if label3_list[0] == label1:
        if list(set(label3_list))[1] == label2:
            label3 = list(set(label3_list))[2]
        else:
            label3 = list(set(label3_list))[1]
    elif label3_list[0] == label2:
        if list(set(label3_list))[1] == label1:
            label3 = list(set(label3_list))[2]
        else:
            label3 = list(set(label3_list))[1]
    else:
        label3 = label3_list[0]
    labels_pred = []
    for i in label_pred:
        if i == 0:
            labels_pred.append(label1)
        if i == 1:
            labels_pred.append(label2)
        if i == 2:
            labels_pred.append(label3)
    acc = ((labels_pred == labels_true).sum())/len(labels_pred)
    return acc, labels_pred


def get_true_label(data_loader):
    """
    获取真实label的文件，方便后续调用
    """
    labels_true = []
    for data in data_loader:
        inputs, label = data
        print(label)
        labels_true.append(label[0].item())
    np.save('label_true.npy', labels_true)
    print(len(labels_true))
    print(labels_true)
