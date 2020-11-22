import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import os

import evaluation.base_evaluation.base_evaluation_config as evaluation_config
import file_process.values_save_and_load as values_save_and_load


def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax


def ROC_multi_main(labels_true_path, labels_pred_path, present_path, class_number, source_dir, each_class_plot=False):
    # 将标签二值化
    classes = []
    for i in range(class_number):
        classes.append(i)
    y = values_save_and_load.load_values(labels_true_path)
    # y_pre = values_save_and_load.load_values(labels_pred_path)
    y1 = []
    for each in y:
        y1.append(each.cpu().numpy())
    # y_pre = label_binarize(y_pre, classes=classes)
    y = label_binarize(y1, classes=classes)
    n_classes = class_number
    y_score = values_save_and_load.load_values(present_path)
    for index, i in enumerate(y_score):
        y_score[index] = softmax(i)
    # y_score = -np.sort(-y_score, axis=1)
    fpr, tpr, _ = roc_curve(y.ravel(), y_score.ravel())
    roc_auc_micro = roc_auc_score(y, y_score, average='micro')

    # Plot all ROC curves
    lw = 2
    plt.figure()

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(n_classes), colors):
    plt.plot(fpr, tpr, color='blue', lw=lw,
             label='AUC = %0.2f' % roc_auc_micro)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC plot')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(source_dir, 'ROC_plot.pdf'))

    # if each_class_plot:
    #     for i, color in zip(range(n_classes), colors):
    #         plt.figure()
    #         plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #                  label='ROC curve of class {0} (area = {1:0.2f})'
    #                  ''.format(i, roc_auc[i]))
    #
    #         plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    #         plt.xlim([0.0, 1.0])
    #         plt.ylim([0.0, 1.05])
    #         plt.xlabel('False Positive Rate')
    #         plt.ylabel('True Positive Rate')
    #         plt.title('Some extension of Receiver operating characteristic to multi-class')
    #         plt.legend(loc="lower right")
    #         plt.savefig(os.path.join(source_dir, 'ROC_plot{}.pdf'.format(i)))

    print('ROC plot has finished!')


if __name__ == '__main__':
    # source_dir_list = [r'D:\My_Code\mengxiangjie\My_Code_Framework\evaluation\data_and_model\NCGCN']
    # # for source_dir in evaluation_config.source_dir_list:
    # for source_dir in source_dir_list:
        y = r'D:\My_Code\mengxiangjie\My_Code_Framework\evaluation\data_and_model\NCGCN\labels_true.npy'
        prob = r'D:\My_Code\mengxiangjie\My_Code_Framework\evaluation\data_and_model\NCGCN\present.npy'
        y_pred = r'D:\My_Code\mengxiangjie\My_Code_Framework\evaluation\data_and_model\NCGCN\labels_pred.npy'
        source_dir = r'D:\My_Code\mengxiangjie\My_Code_Framework\evaluation\data_and_model\NCGCN'
        ROC_multi_main(source_dir=source_dir,
                       labels_true_path=y,
                       labels_pred_path=y_pred,
                       present_path=prob,
                       # class_number=evaluation_config.class_number,
                       class_number=6,
                       each_class_plot=evaluation_config.each_class_plot)
