import sklearn.metrics
import os
import numpy as np

import evaluation.base_evaluation.base_evaluation_config as evaluation_config
import file_process.values_save_and_load as values_save_and_load


def evaluation_value_main(labels_true_path, labels_pred_path, class_names, source_dir):
    labels = values_save_and_load.load_values(labels_true_path)
    predicted = values_save_and_load.load_values(labels_pred_path)
    labels = np.array(labels, dtype=np.int)
    predicted = np.array(predicted, dtype=np.int)
    report1 = sklearn.metrics.classification_report(labels, predicted, target_names=class_names, digits=4, output_dict=True)
    report2 = sklearn.metrics.classification_report(labels, predicted, target_names=class_names, digits=4, output_dict=False)
    NMI = sklearn.metrics.normalized_mutual_info_score(labels, predicted)
    FMI = sklearn.metrics.fowlkes_mallows_score(labels, predicted)
    acc = report1['accuracy']
    precision = report1['macro avg']['precision']
    recall = report1['macro avg']['recall']
    f1_score = report1['macro avg']['f1-score']
    the_list = [report2,
                '\n\naccuracy\tprecision\trecall\tf1-score\n',
                str(int(acc * 1e+4) / 1e+4) + '\t' + str(int(precision * 1e+4) / 1e+4) + '\t' +
                str(int(recall * 1e+4) / 1e+4) + '\t' + str(int(f1_score * 1e+4) / 1e+4),
                'NMI:' + str(NMI),
                'FMI:' + str(FMI)]
    values_save_and_load.save_txt_file(os.path.join(source_dir, 'evaluation_value.txt'), the_list, True, 'w')
    print('Evaluation value has finished!')


if __name__ == '__main__':
    labels_true_path = r'D:\My_Code\mengxiangjie\My_Code_Framework\malaria_evaluation\14_91.0\labels.npy'
    labels_pred_path = r'D:\My_Code\mengxiangjie\My_Code_Framework\malaria_evaluation\14_91.0\predicted.npy'
    source_dir = r'D:\My_Code\mengxiangjie\My_Code_Framework\malaria_evaluation'
    # class_names = ['gametocyte', 'leukocyte', 'red_blood_cell', 'ring', 'schizont', 'trophozoite']
    class_names = ['1', '2']
    # for source_dir in evaluation_config.source_dir_list:
    #     labels_true_path = os.path.join(source_dir, evaluation_config.labels_true_name)
    #     labels_pred_path = os.path.join(source_dir, evaluation_config.labels_pred_name)
    evaluation_value_main(source_dir=source_dir,
                          labels_true_path=labels_true_path,
                          labels_pred_path=labels_pred_path,
                          class_names=evaluation_config.class_names)


