import xml.etree.ElementTree as et
import json
import os


def one_convert(json_path_train, json_path_train_save, label):
    root1 = et.parse(json_path_train)
    path = os.path.join(json_path_train_save, label + '.json')
    f = open(path, 'a', encoding="utf-8")
    all_list = []
    for each in root1.getiterator("object"):
        temp_dict = each.attrib
        child_node = each.getchildren()
        temp_dict[child_node[0].tag] = child_node[0].text  # tag获取节点名,text获取节点值
        for bnd_box in each.getiterator("xmax"):
            temp_dict['x1'] = bnd_box.text
        for bnd_box in each.getiterator("ymax"):
            temp_dict['y1'] = bnd_box.text
        for bnd_box in each.getiterator("xmin"):
            temp_dict['x2'] = bnd_box.text
        for bnd_box in each.getiterator("ymin"):
            temp_dict['y2'] = bnd_box.text
        all_list.append(temp_dict)
    all_dict = {"labels": all_list}
    temp_json = json.dumps(all_dict, ensure_ascii=False)
    print(temp_json)
    f.write(temp_json + '\n')
    f.close()





