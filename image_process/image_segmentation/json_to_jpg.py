import os
import json
from PIL import Image


def json_to_jpg(i, img_path, save_path, json_path_train, label):
    with open(json_path_train, 'r') as json_file:
        data = json.load(json_file)
        item = data
        print(item)
        dir_path = img_path
        for j in range(0, len(item['labels'])):
            # the coordinate of the cell
            ymin = float(item['labels'][j]['y2'])
            xmin = float(item['labels'][j]['x2'])
            ymax = float(item['labels'][j]['y1'])
            xmax = float(item['labels'][j]['x1'])
            ca = item['labels'][j]['name']
            img = Image.open(dir_path)
            box1 = (xmin, ymin, xmax, ymax)  # 设置左、上、右、下的像素
            image1 = img.crop(box1)  # 图像裁剪
            print("creating the cell {} of the {} picture".format(i, j))
            if os.path.exists(save_path + str(label)) is not True:
                os.makedirs(save_path + str(label))
            image1.save(os.path.join(save_path + str(label), 'a{}_{}'.format(i, j) + '.jpg'))


if __name__ == '__main__':
    json_to_jpg(1, './data/bingoner_img/1.bmp', './data/processed_image/', './data/bingoner/1.json', 'label')


