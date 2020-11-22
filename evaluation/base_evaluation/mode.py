import numpy as np
import matplotlib.pyplot as plt


def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax


y = r'D:\My_Code\mengxiangjie\My_Code_Framework\evaluation\data_and_model\NCGCN\labels_true.npy'
prob = r'D:\My_Code\mengxiangjie\My_Code_Framework\evaluation\data_and_model\NCGCN\present.npy'

labels_true_raw = np.load(y, allow_pickle=True)
presents_raw = np.load(prob, allow_pickle=True)
presents = []
labels_true = []

for label in labels_true_raw:
    if label == 0:
        labels_true.append(1)
    else:
        labels_true.append(0)

for i in presents_raw:
    i_new = softmax(i)
    presents.append(np.max(i_new))

presents = np.array(presents)
labels_true = np.array(labels_true)

arg = np.argsort(presents, kind="mergesort")[::-1]
presents = presents[arg]
labels_true = labels_true[arg]

threshold = []
start = int(presents[arg[0]] * 100)
end = int(presents[arg[-1]] * 100)
for i in range(start, end):
    threshold.append(1 / 100 * i)

fp_list = []
tp_list = []
fn_list = []
tn_list = []
for thre in threshold:
    fp = 0
    tp = 0
    fn = 0
    tn = 0
    for index, present in enumerate(presents):
        if present > thre and labels_true[index] == 1:
            tp += 1
        elif present < thre and labels_true[index] == 1:
            fn += 1
        elif present > thre and labels_true[index] == 0:
            fp += 1
        elif present < thre and labels_true[index] == 0:
            tn += 1
    fp_list.append(fp)
    tp_list.append(tp)
    fn_list.append(fn)
    tn_list.append(tn)


fp_arr = np.array(fp_list)
fn_arr = np.array(fn_list)
tp_arr = np.array(tp_list)
tn_arr = np.array(tn_list)

fpr = tp_arr / (tp_arr + fn_arr)
tpr = fp_arr / (fp_arr + tn_arr)

lw = 2
plt.figure(figsize=(6*1.2, 6))
plt.plot(fpr, tpr, color='blue',
         lw=lw)  # 假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC plot')
plt.legend(loc="lower right")
# plt.savefig('ROC{}.pdf'.format(num))
plt.show()



