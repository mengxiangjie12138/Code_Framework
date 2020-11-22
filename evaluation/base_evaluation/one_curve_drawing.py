import matplotlib.pyplot as plt
import os


def train_and_test_value_plot(y1, y2, save_dir):
    plt.figure(figsize=(6 * 1.2, 6))
    plt.bar(range(len(y2)), y2, color='blue', tick_label=y1)
    # plt.xlim(0, len(y1))
    x_label = 'Learning Rate'
    plt.xlabel(x_label)
    plt.ylabel('Accuracy')
    # plt.legend(loc='upper right')

    plt.savefig(os.path.join(save_dir, x_label + '.pdf'))


if __name__ == '__main__':
    # y1 = [18, 34, 50]
    # y2 = [64.83, 55.50, 94.17]
    # y1 = [1, 2, 3, 4, 5]
    # y2 = [88.33, 94.17, 86.67, 72.00, 82.00]
    # y1 = [8, 9, 10, 11, 12]
    # y2 = [92, 91, 94.17, 92.67, 91.83]
    y1 = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    y2 = [65.83, 94.17, 89.6, 88.6, 75.16]
    save_dir = r'D:\My_Code\mengxiangjie\My_Code_Framework\evaluation\data_and_model\NCGCN'
    train_and_test_value_plot(y1, y2, save_dir)















