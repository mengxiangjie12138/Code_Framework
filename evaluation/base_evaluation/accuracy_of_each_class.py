import sklearn.metrics
import numpy as np
import os

import evaluation.base_evaluation.base_evaluation_config as evaluation_config
import file_process.values_save_and_load as values_save_and_load


def get_acc_of_each_class(labels_true, labels_pred, class_names, source_dir):
    cm = sklearn.metrics.confusion_matrix(labels_true, labels_pred)
    diag_list = np.diag(cm)
    each_class_number = np.sum(cm, axis=1)
    acc_list = []
    for i in range(len(class_names)):
        class_acc = diag_list[i] / each_class_number[i]
        acc_list.append(class_names[i] + '\t' + str(class_acc) + '\n')
    values_save_and_load.save_txt_file(os.path.join(source_dir, 'each_class_accuracy.txt'), acc_list, True, 'w')
    print('Acc of each class has finished!')


def acc_of_each_class_main(labels_true_path, labels_pred_path, class_names, source_dir):
    labels_true = values_save_and_load.load_values(labels_true_path)
    labels_pred = values_save_and_load.load_values(labels_pred_path)
    labels_true = np.array(labels_true, dtype=np.int)
    labels_pred = np.array(labels_pred, dtype=np.int)
    # print(labels_pred)
    get_acc_of_each_class(labels_true, labels_pred, class_names, source_dir)


if __name__ == '__main__':
    # for source_dir in evaluation_config.source_dir_list:
        # labels_true_path = os.path.join(source_dir, evaluation_config.labels_true_name)
        # labels_pred_path = os.path.join(source_dir, evaluation_config.labels_pred_name)
        labels_true_path = r'D:\My_Code\mengxiangjie\My_Code_Framework\evaluation\data_and_model\NCGCN\labels_true.npy'
        labels_pred_path = r'D:\My_Code\mengxiangjie\My_Code_Framework\evaluation\data_and_model\NCGCN\labels_pred.npy'
        source_dir = r'D:\My_Code\mengxiangjie\My_Code_Framework\evaluation\data_and_model\NCGCN'
        class_names = ['gametocyte', 'leukocyte', 'red_blood_cell', 'ring', 'schizont', 'trophozoite']
        acc_of_each_class_main(labels_true_path=labels_true_path,
                               labels_pred_path=labels_pred_path,
                               class_names=evaluation_config.class_names,
                               source_dir=source_dir)












