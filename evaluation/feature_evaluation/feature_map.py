import os
import PIL.Image as Image
import torch
from torchvision.utils import save_image
from networks_and_related_funs.ClassicNetwork.ResNet import ResNet50

import feature_extractor.feature_config as feature_config


def feature_map(image, image_name, net, save_dir, use_cuda=True):
    if use_cuda:
        image = feature_config.transform_test(image).cuda()
    else:
        image = feature_config.transform_test(image)
    image = image.unsqueeze(0)
    output = net.conv1(image)[0].data
    for i in range(len(output)):
        if os.path.exists(os.path.join(save_dir, 'feature_map', 'feature_map')) is not True:
            os.makedirs(os.path.join(save_dir, 'feature_map', 'feature_map'))
        if os.path.exists(os.path.join(save_dir, 'feature_map', 'feature_map', image_name)) is not True:
            os.makedirs(os.path.join(save_dir, 'feature_map', 'feature_map', image_name))
        save_image(output[i], os.path.join(save_dir, 'feature_map', 'feature_map', image_name, '{}.png'.format(i)))


def feature_map_main(image_dir, net, save_dir, use_cuda):
    images_list = os.listdir(image_dir)
    for image_name in images_list:
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        feature_map(image, image_name, net, save_dir, use_cuda)


if __name__ == '__main__':
    source_dir = 'test'
    net_name = 'net.pkl'
    image_dir = os.path.join(source_dir, 'feature_map', 'images')
    ResNet = torch.load(os.path.join(source_dir, net_name))
    feature_map_main(image_dir=image_dir,
                     net=ResNet,
                     save_dir=source_dir,
                     use_cuda=True)







