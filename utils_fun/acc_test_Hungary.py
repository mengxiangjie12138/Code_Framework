import numpy as np
from scipy.optimize import linear_sum_assignment


def cluster_acc(y_true, y_pred, class_number):  # 聚类精度  真正标签与预测标签
    cnt_mtx = np.zeros([class_number, class_number])

    # fill in matrix
    for i in range(len(y_true)):
        cnt_mtx[int(y_pred[i]), int(y_true[i])] += 1

    # find optimal permutation
    row_ind, col_ind = linear_sum_assignment(-cnt_mtx)

    # compute error
    acc = cnt_mtx[row_ind, col_ind].sum() / cnt_mtx.sum()

    labels_pred = []
    for index, label in enumerate(y_pred):
        target_label = col_ind[label]
        # print('label', label)
        # print('target', target_label)
        # print('true', y_true[index])
        labels_pred.append(target_label)
    # print(labels_pred[:10])
    # print(list(y_true)[:10])

    return acc, list(y_true), labels_pred


if __name__ == '__main__':
    labels_true = np.load(r'D:\My_Code\mengxiangjie\labels_pred_21.npy')
    labels_pred = np.load(r'D:\My_Code\mengxiangjie\labels_true.npy')
    cluster_acc(labels_true, labels_pred, 10)


