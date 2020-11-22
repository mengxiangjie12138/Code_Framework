import os
import numpy as np


def save_txt_file(path, str_or_list, if_list=False, mode='a'):
    if if_list is True:
        f = open(path, mode)
        for i in str_or_list :
            f.write(str(i) + '\n')
        f.close()
    else:
        f = open(path, mode)
        f.write(str_or_list + '\n')
        f.close()


def load_values(path):
    mode = path.split('.')[-1]
    if mode == 'npy':
        values = np.load(path, allow_pickle=True)
    elif mode == 'txt':
        values = []
        f = open(path, 'r')
        for line in f:
            values.append(float(line))
    else:
        raise Exception('Invalid suffix!')
    return values








