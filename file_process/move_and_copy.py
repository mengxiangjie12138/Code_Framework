import os
import random
import shutil


def copy_file(file_dir, tar_dir, ratio_or_number, one=False):
    file_dir = file_dir + '/'
    tar_dir = tar_dir + '/'
    path_dir = os.listdir(file_dir)
    if ratio_or_number < 1:
        sample = random.sample(path_dir, int(len(path_dir) * ratio_or_number))
    elif ratio_or_number > 1:
        sample = random.sample(path_dir, ratio_or_number)
    elif ratio_or_number == 1 and one is False:
        sample = random.sample(path_dir, int(len(path_dir) * ratio_or_number))
    else:
        sample = random.sample(path_dir, ratio_or_number)
    for name in sample:
        shutil.copyfile(file_dir + name, tar_dir + name)
        print(tar_dir + name)


def move_file(file_dir, tar_dir, ratio_or_number, one=False):
    file_dir = file_dir + '/'
    path_dir = os.listdir(file_dir)
    if ratio_or_number < 1:
        sample = random.sample(path_dir, int(len(path_dir) * ratio_or_number))
    elif ratio_or_number > 1:
        sample = random.sample(path_dir, ratio_or_number)
    elif ratio_or_number == 1 and one is False:
        sample = random.sample(path_dir, int(len(path_dir) * ratio_or_number))
    else:
        sample = random.sample(path_dir, ratio_or_number)
    for name in sample:
        shutil.move(file_dir + name, tar_dir)
        print(tar_dir + '/' + name)


def process(path_list, raw_file_dir, raw_tar_dir, mode='copy', ratio_or_number=1, one=False):
    if os.path.exists(raw_tar_dir) is not True:
        os.makedirs(raw_tar_dir)
    for path_name in path_list:
        file_dir = os.path.join(raw_file_dir, path_name)
        tar_dir = os.path.join(raw_tar_dir, path_name)
        if os.path.exists(tar_dir) is not True:
            os.makedirs(tar_dir)
        if mode == 'move':
            move_file(file_dir, tar_dir, ratio_or_number, one)
        if mode == 'copy':
            copy_file(file_dir, tar_dir, ratio_or_number, one)


def process_one(file_dir, tar_dir, mode='copy', ratio_or_number=1, one=False):
    if os.path.exists(tar_dir) is not True:
        os.makedirs(tar_dir)
    if mode == 'move':
        move_file(file_dir, tar_dir, ratio_or_number, one)
    if mode == 'copy':
        copy_file(file_dir, tar_dir, ratio_or_number, one)






















