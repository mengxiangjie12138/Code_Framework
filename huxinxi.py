import numpy as np


def get_p(p1, p2):
    p = np.zeros([2, 2])
    p[0, 0] = p1[0] * p2[0]
    p[0, 1] = p1[0] * p2[1]
    p[1, 0] = p1[1] * p2[0]
    p[1, 1] = p1[1] * p2[1]
    p = (p + p.T) / 2
    return p


def get_huxinxi(p):
    s1 = np.sum(p, axis=1)
    s2 = np.sum(p, axis=0)
    huxinxi = 0
    h_x = 0
    for i in range(2):
        for j in range(2):
            huxinxi += p[i, j] * np.log2(p[i, j] / (s1[i] * s2[j]))
    for i in range(2):
        h_x += s1[i] * np.log2(s1[i]) * (-1)
    return huxinxi, h_x


for i in range(8):
    p1 = np.array([0.5, 0.5])
    p2 = np.array([0.4, 0.6])
    alpha = i * 0.05
    p1[0] -= alpha
    p2[0] -= alpha
    p1[1] += alpha
    p2[1] += alpha
    p = get_p(p1, p2)
    huxinxi, h_x = get_huxinxi(p)
    print('互信息：', huxinxi, '熵：', h_x, '条件熵：', h_x - huxinxi)





















