import os
import json
import cv2
import sys
from PIL import Image


def get_class1():
    json_path = 'training.json'
    with open(json_path, 'r') as json_file:
        for line in json_file:
            data = json.loads(line)
            for i in range(len(data)):
                classes_num = 0
                item = data[i]
                dir_path = item['image']['pathname']
                classes_name = ['gametocyte', 'leukocyte', 'red blood cell', 'ring', 'schizont', 'trophozoite']
                for classes in classes_name:
                    for j in range(0, len(item['objects'])):
                        ca = item['objects'][j]['category']
                        if classes == ca:
                            classes_num += 1
                            break
                if classes_num >= 4:
                    print(i)


def get_class2(i, raw_path_dir, save_dir_path=None, raw_size=120):
    json_path = 'training.json'
    if save_dir_path is None:
        if os.path.exists(str(i)) is not True:
            os.makedirs(str(i))
        save_dir_path = str(i)
    with open(json_path, 'r') as json_file:
        for line in json_file:
            data = json.loads(line)
            item = data[i]
            dir_path = item['image']['pathname']
            classes_name = ['gametocyte', 'leukocyte', 'red blood cell', 'ring', 'schizont', 'trophozoite']
            print(dir_path)
            for classes in classes_name:
                for j in range(0, len(item['objects'])):
                    ymin = item['objects'][j]['bounding_box']['minimum']['r']
                    xmin = item['objects'][j]['bounding_box']['minimum']['c']
                    ymax = item['objects'][j]['bounding_box']['maximum']['r']
                    xmax = item['objects'][j]['bounding_box']['maximum']['c']
                    ca = item['objects'][j]['category']
                    if classes == ca:
                        img = Image.open(raw_path_dir + dir_path)
                        if os.path.exists(os.path.join(save_dir_path, 'raw.png')) is not True:
                            img1 = img.resize((raw_size, raw_size), Image.ANTIALIAS)
                            img1.save(os.path.join(save_dir_path, 'raw.png'))
                        box = (xmin, ymin, xmax, ymax)  # 设置左、上、右、下的像素
                        image = img.crop(box)  # 图像裁剪
                        image = image.resize((raw_size//4 - 5, raw_size//4 - 5), Image.ANTIALIAS)
                        image.save(os.path.join(save_dir_path, ca + '.png'))
                        print('class_name:', ca, 'ymin:', ymin, 'xmin:', xmin, 'ymax:', ymax, 'xmax:', xmax)
                        break


if __name__ == '__main__':
    raw_path_dir = 'malaria/malaria'
    get_class2(356, raw_path_dir, raw_size=600)

# 56,220,335,342
