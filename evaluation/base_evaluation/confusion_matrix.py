import sklearn.metrics
import matplotlib.pyplot as plt
import os
import numpy as np

import evaluation.base_evaluation.base_evaluation_config as evaluation_config
import file_process.values_save_and_load as values_save_and_load


def confusion_matrix(labels_true, labels_pred, path):
    confusion_matrix = sklearn.metrics.confusion_matrix(labels_true, labels_pred)
    plt.matshow(confusion_matrix)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path)
    return confusion_matrix


def confusion_matrix_main(labels_true_path, labels_pred_path, save_dir):
    labels = values_save_and_load.load_values(labels_true_path)
    predicted = values_save_and_load.load_values(labels_pred_path)
    labels = np.array(labels, dtype=np.int)
    predicted = np.array(predicted, dtype=np.int)
    cm = confusion_matrix(labels, predicted, os.path.join(save_dir, 'confusion_matrix.pdf'))
    values_save_and_load.save_txt_file(os.path.join(save_dir, 'confusion_matrix.txt'), cm, True, 'w')
    print('Confusion matrix has finished!')


if __name__ == '__main__':
    labels_true_path = r'D:\My_Code\mengxiangjie\My_Code_Framework\evaluation\data_and_model\NCGCN\labels_true.npy'
    labels_pred_path = r'D:\My_Code\mengxiangjie\My_Code_Framework\evaluation\data_and_model\NCGCN\labels_pred.npy'
    source_dir = r'D:\My_Code\mengxiangjie\My_Code_Framework\evaluation\data_and_model\NCGCN'
    class_names = ['gametocyte', 'leukocyte', 'red_blood_cell', 'ring', 'schizont', 'trophozoite']
    # for source_dir in evaluation_config.source_dir_list:
    #     labels_true_path = os.path.join(source_dir, evaluation_config.labels_true_name)
    #     labels_pred_path = os.path.join(source_dir, evaluation_config.labels_pred_name)
    confusion_matrix_main(save_dir=source_dir,
                          labels_true_path=labels_true_path,
                          labels_pred_path=labels_pred_path)




