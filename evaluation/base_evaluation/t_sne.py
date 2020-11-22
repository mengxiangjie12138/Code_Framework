import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import os

import evaluation.base_evaluation.base_evaluation_config as evaluation_config
import file_process.values_save_and_load as values_save_and_load


def get_shuffle_classes(labels, features, num, class_num):
    names = locals()
    range_raw = len(labels) // class_num
    for raw in range(class_num):
        raw_n = random.sample(range(range_raw * raw, range_raw * (raw + 1)), num)
        exec('raw{} = raw_n'.format(raw))

    labels_new = []
    features_new = []
    for i in range(class_num):
        raw_n = names.get('raw{}'.format(i))
        for j in raw_n:
            labels_new.append(labels[j])
            features_new.append(features[j])
    return labels_new, features_new


def t_sne_plot(plot_path, features, labels, classes, class_num):
    tsne = TSNE(n_components=2, learning_rate=100).fit_transform(features)
    names = locals()  # 获取exec动态命名的变量
    plt.figure(figsize=(7, 6))

    c_list = ['b', 'orange', 'navy', 'y', 'm', 'c', 'k']
    for i in range(0, class_num):
        # 定义变量
        exec('tsnex{} = []'.format(i))
        exec('tsney{} = []'.format(i))
        # 添加tsne
        for index, j in enumerate(labels):
            if j == i:
                exec('tsnex{}.append(tsne[{}, {}])'.format(j, index, 0))
                exec('tsney{}.append(tsne[{}, {}])'.format(j, index, 1))

        tsnex = names.get('tsnex{}'.format(i))  # tsnex_list
        tsney = names.get('tsney{}'.format(i))  # tsney_list
        c = c_list[i]
        plt.scatter(tsnex, tsney, label=classes[i], c=c, s=10)

    plt.legend(loc="upper left")
    plt.xlabel(' ')
    plt.ylabel(' ')
    plt.title('t-sne')

    plt.savefig(plot_path)


def t_sne_main(labels_true_path, features_path, class_number, class_names, source_dir, balance=False, each_class_point=100):
    features = values_save_and_load.load_values(features_path)
    labels = values_save_and_load.load_values(labels_true_path)
    labels = np.array(labels, dtype=np.int)
    if balance:
        labels_new, features_new = get_shuffle_classes(labels, features, each_class_point, class_number)
        labels, features = labels_new, features_new
    data = np.reshape(features, (len(features), -1))
    t_sne_plot(os.path.join(source_dir, 't_sne.pdf'), data, labels, class_names, class_number)
    print('t_sne plot has finished!')


if __name__ == '__main__':
    # for source_dir in evaluation_config.source_dir_list:
        labels_true_path = r'D:\My_Code\mengxiangjie\My_Code_Framework\evaluation\data_and_model\NCGCN\labels_true.npy'
        labels_pred_path = r'D:\My_Code\mengxiangjie\My_Code_Framework\evaluation\data_and_model\NCGCN\labels_pred.npy'
        features_path = r'D:\My_Code\mengxiangjie\My_Code_Framework\evaluation\data_and_model\NCGCN\features.npy'
        source_dir = r'D:\My_Code\mengxiangjie\My_Code_Framework\evaluation\data_and_model\NCGCN'
        class_names = ['gametocyte', 'leukocyte', 'red_blood_cell', 'ring', 'schizont', 'trophozoite']
        # labels_true_path = os.path.join(source_dir, evaluation_config.labels_true_name)
        # features_path = os.path.join(source_dir, evaluation_config.features_name)
        t_sne_main(source_dir=source_dir,
                   labels_true_path=labels_true_path,
                   features_path=features_path,
                   class_number=evaluation_config.class_number,
                   class_names=evaluation_config.class_names,
                   balance=evaluation_config.tsne_balance,
                   each_class_point=evaluation_config.tsne_point_num)


