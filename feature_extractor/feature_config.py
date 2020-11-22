import torchvision.transforms as transforms

acc_fluctuation_limit = 3
exit_limit = 5

save_dir = 'Model_and_Log'
model_name = 'ResNet'

acc_file_name = 'acc.txt'
loss_file_name = 'loss.txt'
precision_file_name = 'precision.txt'
recall_file_name = 'recall.txt'
f1_score_file_name = 'f1-score.txt'
acc_test_file_name = 'acc_test.txt'
loss_test_file_name = 'loss_test.txt'
precision_test_file_name = 'precision_test.txt'
recall_test_file_name = 'recall_test.txt'
f1_score_test_file_name = 'f1-score_test.txt'
time_file_name = 'time.txt'
evaluation_name = 'evaluation_value_test.txt'

dataset_dir = r'E:\MXJ_data\malaria'
train_dataset_name = 'train'
test_dataset_name = 'test'


cuda_num = 0
# class_names = ['Babesia', 'LiShiMan', 'Malaria', 'maodichong', 'Toxoplasma', 'ZhuiXingChong', 'RB cell', 'White Cell 100X']
class_names = ['Parasitized', 'Uninfected']
# path_list = ['Babesia', 'LiShiMan', 'Malaria', 'maodichong', 'Toxoplasma', 'ZhuiXingChong']
class_number = len(class_names)

# 超参数设置
best_acc_lower_limit = 0
epoch = 1000
pre_epoch = 0
train_batch_size = 70
test_batch_size = 70
LR = 2e-6

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])














