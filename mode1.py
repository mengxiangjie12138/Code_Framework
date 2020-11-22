import numpy as np
import csv
import matplotlib.pyplot as plt

csv_reader = csv.reader(open('1.csv', encoding='utf-8'))
M_list = []
N_list = []
X_list = []
Y_list = []
OH_list = []
HorY_list = []
OV_list = []
VerX_list = []
for row in csv_reader:
    data = row[0].split('\t')
    M_list.append(eval(data[0])-1)
    N_list.append(eval(data[1])-1)
    OH_list.append(eval(data[2]))
    HorY_list.append(eval(data[3]))
    OV_list.append(eval(data[4]))
    VerX_list.append(eval(data[5]))

X = np.zeros([28, 28])
Y = np.zeros([28, 28])
M = np.zeros([28, 28])
N = np.zeros([28, 28])
OH = np.zeros([28, 28])
HorY = np.zeros([28, 28])
OV = np.zeros([28, 28])
VerX = np.zeros([28, 28])
OH_new_list = []
OV_new_list = []


for n in range(28):
    for m in range(28):
        pos = n*28+m
        OH[m, n] = OH_list[pos]
        HorY[m, n] = HorY_list[pos]
        OV[m, n] = OV_list[pos]
        VerX[m, n] = VerX_list[pos]

# X[m, n+1] = 12000 + X[m, n] - OH[m, n]
# X[m+1, n] = X[m, n] + VerX[m, n]
# Y[m, n+1] = HorY[m, n] + Y[m, n]
# Y[m+1, n] = 12000 + Y[m, n] - OV[m, n]
# OH(m,n) = 12k - (X(m,n+1)-X(m,n))
# OV(m,n) = 12k - (Y(m+1,n)-Y(m,n))

# OH_new_list.append(0)
# OV_new_list.append(0)
for n in range(28):
    for m in range(28):
        if m != 27 and n != 27:
            X[m, n+1] = 12000 + X[m, n] - OH[m, n]
            X[m+1, n] = X[m, n] + VerX[m, n]
            Y[m, n+1] = HorY[m, n] + Y[m, n]
            Y[m+1, n] = 12000 + Y[m, n] - OV[m, n]

            OH_new_list.append(OH[m, n] / 12000 * 100)
            OV_new_list.append(OV[m, n] / 12000 * 100)

        elif m == 27 and n != 27:
            X[m, n + 1] = 12000 + X[m, n] - OH[m, n]
            Y[m, n + 1] = HorY[m, n] + Y[m, n]
            OH_new_list.append(OH[m, n] / 12000 * 100)
            OV_new_list.append(OV[m, n] / 12000 * 100)
        elif m != 27 and n == 27:
            X[m + 1, n] = X[m, n] + VerX[m, n]
            Y[m + 1, n] = 12000 + Y[m, n] - OV[m, n]
            OH_new_list.append(OH[m, n] / 12000 * 100)
            OV_new_list.append(OV[m, n] / 12000 * 100)
        elif m == n == 27:
            pass
        X_list.append(X[m, n])
        Y_list.append(Y[m, n])

# 数据导出
# csv_data = []
# csv_file = open('data.csv', 'w', encoding='utf-8')
# writer = csv.writer(csv_file)
# writer.writerow(['Row', 'Col', 'OH', 'Hor-Y', 'OV', 'Ver-X', 'X', 'Y'])
# for i in range(784):
#     csv_data.append((i % 28 + 1, i // 28 + 1, OH_list[i], HorY_list[i], OV_list[i], VerX_list[i], X_list[i], Y_list[i]))
# writer.writerows(csv_data)
# csv_file.close()

# 可视化
plt.bar(range(784), np.array(OH_new_list))
plt.title('OH')
plt.savefig('OH.png')
plt.bar(range(784), np.array(OV_new_list))
plt.title('OV')
plt.savefig('OV.png')












