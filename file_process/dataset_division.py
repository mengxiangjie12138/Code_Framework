import os
from file_process.move_and_copy import process, process_one


def dataset_reduction(train_path, test_path, class_names):
    for dir in class_names:
        file_dir = os.path.join(test_path, dir)
        tar_dir = os.path.join(train_path, dir)
        number = len(os.listdir(file_dir))
        process_one(file_dir, tar_dir, mode='move', ratio_or_number=number)


if __name__ == '__main__':
    mode = 'division'  # division or reduction
    # mode = 'reduction'
    ratio_or_number = 1000
    class_names = ['Babesia', 'LiShiMan', 'Malaria', 'maodichong', 'Toxoplasma', 'ZhuiXingChong', 'RB cell', 'White Cell 100X']
    dataset_path = 'dataset_8'
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')
    if mode == 'division':
        process(path_list=class_names,
                raw_file_dir=train_path,
                raw_tar_dir=test_path,
                mode='move',
                ratio_or_number=ratio_or_number,
                one=False)
    elif mode == 'reduction':
        dataset_reduction(train_path=train_path,
                          test_path=test_path,
                          class_names=class_names)
















