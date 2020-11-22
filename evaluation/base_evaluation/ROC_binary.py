import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

import evaluation.base_evaluation.base_evaluation_config as evaluation_config
import file_process.values_save_and_load as values_save_and_load


def ROC_binary_main(labels_true_path, labels_pred_path, present_path, source_dir):
    y_test = values_save_and_load.load_values(labels_true_path)
    y_predicted = values_save_and_load.load_values(labels_pred_path)
    y_score_raw = values_save_and_load.load_values(present_path)
    y_score_new = []
    for i in y_score_raw:
        y_score_new.append(np.max(i))

    fpr, tpr, threshold = roc_curve(y_test, y_score_new, pos_label=y_predicted)
    roc_auc = auc(fpr, tpr)

    lw = 2
    plt.figure(figsize=(6*1.2, 6))
    plt.plot(fpr, tpr, color='red',
             lw=lw, label='AUC = %0.2f' % roc_auc)  # 假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC plot')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(source_dir, 'ROC_plot.pdf'))
    print('ROC plot has finished!')


if __name__ == '__main__':
    for source_dir in evaluation_config.source_dir_list:
        labels_true_path = os.path.join(source_dir, evaluation_config.labels_true_name)
        labels_pred_path = os.path.join(source_dir, evaluation_config.labels_pred_name)
        present_path = os.path.join(source_dir, evaluation_config.present_name)
        ROC_binary_main(source_dir=source_dir,
                        labels_true_path=labels_true_path,
                        labels_pred_path=labels_pred_path,
                        present_path=present_path)

