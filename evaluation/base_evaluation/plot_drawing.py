import matplotlib.pyplot as plt
import os

import file_process.values_save_and_load as values_save_and_load


def train_and_test_value_plot(y1, y2, y_label, save_dir):
    plt.figure(figsize=(6 * 1.2, 6))
    plt.plot(range(len(y1)), y1, '.-', label="Train", color='blue')
    plt.plot(range(len(y2)), y2, '.-', label="Test", color='orange')
    plt.xlim(0, len(y1))
    plt.xlabel('Epoches')
    if y_label == '.acc':
        plot_label = 'Accuracy'
    elif y_label == '.loss':
        plot_label = 'Loss'
    elif y_label == '.precision':
        plot_label = 'Precision'
    elif y_label == '.recall':
        plot_label = 'Recall'
    elif y_label == 'f1-score':
        plot_label = 'F1-score'
    else:
        plot_label = None
    plt.ylabel(plot_label)
    plt.legend(loc='upper right')

    plt.savefig(os.path.join(save_dir, y_label[1:] + '.pdf'))
    print(y_label[1:], 'plot has finished!')


def train_and_test_value_plot_main(y1_path, y2_path, y_label, save_dir):
    y1 = values_save_and_load.load_values(y1_path)
    y2 = values_save_and_load.load_values(y2_path)
    train_and_test_value_plot(y1, y2, y_label, save_dir)


if __name__ == '__main__':
    source_dir = r'D:\My_Code\mengxiangjie\My_Code_Framework\evaluation\data_and_model\NCGCN'
    # tar_names = ['acc.txt', 'precision.txt', 'recall.txt', 'f1_score.txt']
    tar_names = ['Loss.txt']
    for tar_name in tar_names:
        y1_path = os.path.join(source_dir, 'train_' + tar_name)
        y2_path = os.path.join(source_dir, 'test_' + tar_name)
        y_label = ('.' + tar_name.split('.')[-2]).capitalize()
        if os.path.exists(y1_path) and os.path.exists(y2_path):
            train_and_test_value_plot_main(y1_path, y2_path, y_label, source_dir)














