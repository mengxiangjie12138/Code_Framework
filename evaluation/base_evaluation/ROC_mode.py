import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from sklearn import metrics
from sklearn.metrics import roc_curve, auc  # 计算roc和auc

def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax


def ROC_binary_main(labels_true_path, labels_pred_path, present_path):
    fpr_list = []
    tpr_list = []

    for num in range(0, 6):
        y_test = np.load(labels_true_path, allow_pickle=True)
        y_predicted = np.load(labels_pred_path, allow_pickle=True)
        # y_test = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        # y_predicted = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

        y_score_raw = np.load(present_path, allow_pickle=True)
        # y_score_raw = np.random.random(len(y_test))
        y_score_new = []
        y_test_new = []
        y_predicted_new = []
        for y in y_test:
            if y == num:
                y_test_new.append(1)
            else:
                y_test_new.append(0)
        for y_pred in y_predicted:
            if y_pred == num:
                y_predicted_new.append(1)
            else:
                y_predicted_new.append(0)
        for i in y_score_raw:
            i_new = softmax(i)
            y_score_new.append(i_new[num])

        # print(y_predicted_new)
        # print(y_test_new)
        # print(y_score_new)
        # print(np.sum(y_test_new))
        # print(len(y_test_new))
        # print(np.sum(y_predicted_new))
        # print(len(y_predicted_new))

        fpr, tpr, threshold = roc_curve(y_test_new, y_score_new, pos_label=1)
        fpr_list.append(np.mean(fpr))
        tpr_list.append(np.mean(tpr))
        print(fpr)
        # print(fpr)
        # print(tpr)
        # exit()

        roc_auc = auc(fpr, tpr)

        lw = 2
        plt.figure(figsize=(6*1.2, 6))
        plt.plot(fpr, tpr, color='blue',
                 lw=lw, label='AUC = %0.2f' % roc_auc)  # 假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC plot')
        plt.legend(loc="lower right")
        plt.savefig('ROC{}.pdf'.format(num))
        # plt.show()
        print('ROC plot{} has finished!'.format(num))
    y_test = np.load(labels_true_path, allow_pickle=True)
    y_predicted = np.load(labels_pred_path, allow_pickle=True)
    # y_test = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    # y_predicted = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    y_score_new = []
    y_score_raw = np.load(present_path, allow_pickle=True)
    for i in y_score_raw:
        i_new = softmax(i)
        y_score_new.append(np.max(i_new))
    fpr, tpr, threshold = roc_curve(y_test, y_score_new)
    roc_auc = auc(fpr, tpr)
    lw = 2
    plt.figure(figsize=(6 * 1.2, 6))
    plt.plot(fpr, tpr, color='blue',
             lw=lw, label='AUC = %0.2f' % roc_auc)  # 假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC plot')
    plt.legend(loc="lower right")
    plt.savefig('ROC{}.pdf'.format(100))


if __name__ == '__main__':
    y = r'D:\My_Code\mengxiangjie\My_Code_Framework\evaluation\data_and_model\NCGCN\labels_true.npy'
    prob = r'D:\My_Code\mengxiangjie\My_Code_Framework\evaluation\data_and_model\NCGCN\present.npy'
    y_pred = r'D:\My_Code\mengxiangjie\My_Code_Framework\evaluation\data_and_model\NCGCN\labels_pred.npy'
    ROC_binary_main(y, y_pred, prob)










