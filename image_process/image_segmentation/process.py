import os
from image_process.image_segmentation.xml_to_json import one_convert
from image_process.image_segmentation.json_to_jpg import json_to_jpg


json_path_train = 'OtherData/xueyezhuitichong'
json_path_train_save = './json/xueyezhuitichong'
img_path = 'OtherData/血液椎体虫'
save_path = './data/'
label_name = 'xueyezhuitichong'


if __name__ == '__main__':
    if os.path.exists(json_path_train_save) is not True:
        os.makedirs(json_path_train_save)
    if os.path.exists(save_path) is not True:
        os.makedirs(save_path)
    list_xml = os.listdir(json_path_train)
    for list_xml_element in list_xml:
        a = list_xml_element.split('.')[0]
        one_convert(os.path.join(json_path_train, list_xml_element), json_path_train_save, a)
        list_json_element = os.path.join(json_path_train_save, a + '.json')
        img_path_element = os.path.join(img_path, a + '.bmp')
        json_to_jpg(int(a), img_path_element, save_path, list_json_element, label_name)
    print('Job have done')








