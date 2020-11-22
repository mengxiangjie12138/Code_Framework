import os

from evaluation.base_evaluation.confusion_matrix import confusion_matrix_main
from evaluation.base_evaluation.evaluation_value import evaluation_value_main
from evaluation.base_evaluation.ROC_binary import ROC_binary_main
from evaluation.base_evaluation.ROC_multi import ROC_multi_main
from evaluation.base_evaluation.t_sne import t_sne_main
from evaluation.base_evaluation.accuracy_of_each_class import acc_of_each_class_main

import evaluation.base_evaluation.base_evaluation_config as evaluation_config


all_evaluation = ['confusion_matrix', 'evaluation_value', 'ROC', 't_sne', 'accuracy_of_each_class']


def evaluation_process(source_dir,
                       labels_true_name=evaluation_config.labels_true_name,
                       labels_pred_name=evaluation_config.labels_pred_name,
                       present_name=evaluation_config.present_name,
                       features_name=evaluation_config.features_name,
                       evaluation_list=evaluation_config.evaluation_fun_list,
                       class_names=evaluation_config.class_names,
                       class_number=evaluation_config.class_number,
                       each_class_plot=evaluation_config.each_class_plot,
                       tsne_balance=evaluation_config.tsne_balance,
                       tsne_point_num=evaluation_config.tsne_point_num):
    labels_true_path = os.path.join(source_dir, labels_true_name)
    labels_pred_path = os.path.join(source_dir, labels_pred_name)
    present_path = os.path.join(source_dir, present_name)
    features_path = os.path.join(source_dir, features_name)
    for evaluation in evaluation_list:
        if evaluation == 'confusion_matrix':
            confusion_matrix_main(save_dir=source_dir,
                                  labels_true_path=labels_true_path,
                                  labels_pred_path=labels_pred_path)
        elif evaluation == 'evaluation_value':
            evaluation_value_main(source_dir=source_dir,
                                  labels_true_path=labels_true_path,
                                  labels_pred_path=labels_pred_path,
                                  class_names=class_names)
        elif evaluation == 'ROC' and class_number == 2:
            ROC_binary_main(source_dir=source_dir,
                            labels_true_path=labels_true_path,
                            labels_pred_path=labels_pred_path,
                            present_path=present_path)
        elif evaluation == 'ROC' and class_number > 2:
            ROC_multi_main(source_dir=source_dir,
                           labels_true_path=labels_true_path,
                           labels_pred_path=labels_pred_path,
                           present_path=present_path,
                           class_number=class_number,
                           each_class_plot=each_class_plot)
        elif evaluation == 't_sne':
            t_sne_main(source_dir=source_dir,
                       labels_true_path=labels_true_path,
                       features_path=features_path,
                       class_number=class_number,
                       class_names=class_names,
                       balance=tsne_balance,
                       each_class_point=tsne_point_num)
        elif evaluation == 'accuracy_of_each_class':
            acc_of_each_class_main(source_dir=source_dir,
                                   labels_true_path=labels_true_path,
                                   labels_pred_path=labels_pred_path,
                                   class_names=class_names)


if __name__ == '__main__':
    for source_dir in evaluation_config.source_dir_list:
        evaluation_process(source_dir=source_dir)
    print('All finished!')



















