import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import time
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn import metrics

from networks_and_related_funs.ClassicNetwork.ResNet import ResNet50
from feature_extractor import utils_fun, feature_config
import file_process.values_save_and_load as values_save_and_load


def train(save_dir=feature_config.save_dir, model_name=feature_config.model_name, learning_rate=feature_config.LR,
          pre_epoch=feature_config.pre_epoch, end_epoch=feature_config.epoch,
          dataset_dir=feature_config.dataset_dir,
          train_dataset_name=feature_config.train_dataset_name, test_dataset_name=feature_config.test_dataset_name,
          transform_train=feature_config.transform_train, transform_test=feature_config.transform_test,
          train_batch_size=feature_config.train_batch_size, test_batch_size=feature_config.test_batch_size,
          cuda_num=feature_config.cuda_num,
          class_names=feature_config.class_names, class_number=feature_config.class_number,
          best_acc_lower_limit=feature_config.best_acc_lower_limit,
          limit_epoch=False,
          acc_fluctuation_limit=feature_config.acc_fluctuation_limit,
          exit_limit=feature_config.exit_limit):

    model_dir = os.path.join(save_dir, model_name)

    device = torch.device("cuda:{}".format(cuda_num) if torch.cuda.is_available() else "cpu")

    train_set = datasets.ImageFolder(os.path.join(dataset_dir, train_dataset_name), transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)

    test_set = datasets.ImageFolder(os.path.join(dataset_dir, test_dataset_name), transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True)

    net = ResNet50(class_number).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    if os.path.exists(save_dir) is not True:
        os.makedirs(save_dir)
    if os.path.exists(model_dir) is not True:
        os.makedirs(model_dir)
    best_acc = best_acc_lower_limit
    pre_acc = 0
    exit_num = 0
    for epoch in range(pre_epoch, end_epoch):
        since = time.time()
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        labels_list = []
        predicted_list = []
        length = len(train_loader)
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs, _ = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            for label in labels:
                labels_list.append(label.item())
            for pre in predicted:
                predicted_list.append(pre.item())
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()

            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% |time:%.3f'
                  % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total, time.time() - since))
        acc_test, precision, recall, f1_score = utils_fun.get_evaluation_value_each_epoch(labels_list,
                                                                                          predicted_list,
                                                                                          class_names)
        values_save_and_load.save_txt_file(os.path.join(model_dir, feature_config.acc_file_name), str(acc_test), 'a')
        values_save_and_load.save_txt_file(os.path.join(model_dir, feature_config.loss_file_name), str(sum_loss / length), 'a')
        values_save_and_load.save_txt_file(os.path.join(model_dir, feature_config.precision_file_name), str(precision), 'a')
        values_save_and_load.save_txt_file(os.path.join(model_dir, feature_config.recall_file_name), str(recall), 'a')
        values_save_and_load.save_txt_file(os.path.join(model_dir, feature_config.f1_score_file_name), str(f1_score), 'a')

        if epoch % 1 == 0:
            print('start to test')
            with torch.no_grad():
                labels_list = []
                predicted_list = []
                present_list = []
                out1_list = []
                loss = 0

                for data in test_loader:
                    net.eval()
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs, features = net(images)
                    out1_list.append(features.cpu().data.numpy())
                    loss = criterion(outputs, labels)
                    present, predicted = torch.max(outputs.data, 1)

                    for label in labels:
                        labels_list.append(label.item())
                    for pre in predicted:
                        predicted_list.append(pre.item())
                    for pres in outputs:
                        present_list.append(pres.cpu().data.numpy())

                acc_test, precision, recall, f1_score = utils_fun.get_evaluation_value_each_epoch(labels_list,
                                                                                                  predicted_list,
                                                                                                  class_names)
                NMI = metrics.normalized_mutual_info_score(labels_list, predicted_list)
                FMI = metrics.fowlkes_mallows_score(labels_list, predicted_list)
                values_save_and_load.save_txt_file(os.path.join(model_dir, feature_config.acc_test_file_name), str(acc_test), 'a')
                values_save_and_load.save_txt_file(os.path.join(model_dir, feature_config.loss_test_file_name), str(loss.item()), 'a')
                values_save_and_load.save_txt_file(os.path.join(model_dir, feature_config.precision_test_file_name), str(precision), 'a')
                values_save_and_load.save_txt_file(os.path.join(model_dir, feature_config.recall_t nest_file_name), str(recall), 'a')
                values_save_and_load.save_txt_file(os.path.join(model_dir, feature_config.f1_score_test_file_name), str(f1_score), 'a')
                values_save_and_load.save_txt_file(os.path.join(model_dir, 'NMI.txt'), str(NMI), 'a')
                values_save_and_load.save_txt_file(os.path.join(model_dir, 'FMI.txt'), str(FMI), 'a')
                time_consumed = time.time() - since
                print('测试分类准确率为：%.3f%%, time:%.3f' % (100 * acc_test, time_consumed))
                if acc_test >= best_acc:
                    best_acc = acc_test
                    log_dir = os.path.join(model_dir, str(epoch) + '_' + str(100 * acc_test))
                    if os.path.exists(log_dir) is not True:
                        os.makedirs(log_dir)
                    np.save(os.path.join(log_dir, 'labels.npy'), labels_list)
                    np.save(os.path.join(log_dir, 'predicted.npy'), predicted_list)
                    np.save(os.path.join(log_dir, 'present.npy'), present_list)
                    np.save(os.path.join(log_dir, 'features.npy'), np.concatenate(out1_list))
                    evaluation_value = ['loss: ' + str(loss.item()),
                                        'acc: ' + str(acc_test),
                                        'precision: ' + str(precision),
                                        'recall: ' + str(recall),
                                        'f1_score: ' + str(f1_score),
                                        'time: ' + str(time_consumed)]
                    values_save_and_load.save_txt_file(os.path.join(log_dir, feature_config.evaluation_name), evaluation_value, True, 'w')
                    torch.save(net, os.path.join(log_dir, 'model.pkl'))
        # if epoch % 10 == 0:
        #     pre_acc = 100 * best_acc
        # if torch.abs(100 * acc_test - pre_acc) <= acc_fluctuation_limit:
        #     exit_num += 1
        #     if exit_num >= exit_limit and limit_epoch is True:
        #         exit()
        # else:
        #     exit_num = 0
        scheduler.step(epoch)








