import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import time
from torchvision import datasets
from torch.utils.data import DataLoader

from networks_and_related_funs.ClassicNetwork.ResNet import ResNet50
from machine_learning.clustering_algorithm.k_means import MyKMeans
import kmeans_feature_extractor.config as kmeans_features_config
from networks_and_related_funs.loss_fun import KMeansLoss
import kmeans_feature_extractor.utils_fun as utils_fun

data_set = datasets.ImageFolder(os.path.join(kmeans_features_config.dataset_dir, kmeans_features_config.train_dataset_name),
                                 transform=kmeans_features_config.transform_train)
data_loader = DataLoader(data_set, batch_size=1, shuffle=False)


def train(save_dir=kmeans_features_config.save_dir, model_name=kmeans_features_config.model_name, learning_rate=kmeans_features_config.LR,
          pre_epoch=kmeans_features_config.pre_epoch, end_epoch=kmeans_features_config.epoch,
          dataset_dir=kmeans_features_config.dataset_dir,
          train_dataset_name=kmeans_features_config.train_dataset_name,
          transform_train=kmeans_features_config.transform_train,
          train_batch_size=kmeans_features_config.train_batch_size,
          cuda_num=kmeans_features_config.cuda_num,
          class_names=kmeans_features_config.class_names, class_number=kmeans_features_config.class_number,
          best_acc_lower_limit=kmeans_features_config.best_acc_lower_limit,
          limit_epoch=False,
          acc_fluctuation_limit=kmeans_features_config.acc_fluctuation_limit,
          exit_limit=kmeans_features_config.exit_limit):

    model_dir = os.path.join(save_dir, model_name)

    device = torch.device("cuda:{}".format(cuda_num) if torch.cuda.is_available() else "cpu")

    data_set = datasets.ImageFolder(os.path.join(dataset_dir, train_dataset_name), transform=transform_train)
    data_loader = DataLoader(data_set, batch_size=train_batch_size, shuffle=False)

    net = ResNet50(class_number).to(device)

    criterion = KMeansLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    kmeans = MyKMeans(class_number, 0)

    if os.path.exists(save_dir) is not True:
        os.makedirs(save_dir)
    if os.path.exists(model_dir) is not True:
        os.makedirs(model_dir)
    best_acc = best_acc_lower_limit
    pre_acc = 0
    exit_num = 0
    centers = torch.tensor(0., requires_grad=True)
    for epoch in range(pre_epoch, end_epoch):
        since = time.time()
        net.train()
        features_list = []
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs, features = net(inputs)
            if epoch != pre_epoch:
                loss = criterion(features, centers)
                loss.backward()
                optimizer.step()
            features_list.append(features.data.cpu().numpy())
            print(train_batch_size * i)
        features = np.concatenate(features_list, axis=0)
        print(features.shape)
        centers, labels = kmeans(features)
        centers = torch.tensor(centers).to(device)
        acc, pred = utils_fun.test_acc(labels)
        print('acc:', acc)
        f = open('acc_test.txt', 'a')
        f.write(str(acc) + '\n')
        f.close()


if __name__ == '__main__':
    if os.path.exists('label_true.npy') is not True:
        utils_fun.get_true_label(data_loader)
    train()








