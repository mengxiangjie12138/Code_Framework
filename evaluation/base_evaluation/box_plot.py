import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    y1 = [66.16, 67.66, 68.83, 67.16, 66.33]
    y2 = [81.33, 81.0, 83.83, 83.0, 83.33]
    y3 = [85.66, 86.33, 87.83, 87.00, 86.83]
    y4 = [89.16, 90.16, 90.66, 90.33, 88.66]
    y5 = [92.16, 91.83, 94.16, 92.5, 92.7]
    plt.figure(figsize=(6 * 1.2, 6))
    plt.boxplot((y1, y2, y3, y4, y5), positions=[0.2, 0.4, 0.6, 0.8, 1])
    plt.plot([0.2, 0.4, 0.6, 0.8, 1], [y1[3], y2[3], y3[3], y4[3], y5[3]], color='red')
    plt.xlim(0, 1.2)
    x_label = 'Training Size'
    plt.xlabel(x_label)
    plt.ylabel('Accuracy')
    # plt.legend(loc='upper right')

    plt.savefig(os.path.join(r'D:\My_Code\mengxiangjie\My_Code_Framework\evaluation\data_and_model\NCGCN', x_label + '.pdf'))
    plt.show()










