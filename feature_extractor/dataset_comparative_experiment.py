import os

from feature_extractor import train_fun, feature_config
from file_process.move_and_copy import process

if __name__ == '__main__':
    ratios = [0.2, 0.4, 0.6, 0.8]
    for ratio in ratios:
        save_dir = feature_config.save_dir + '-' + str(ratio)
        dataset_dir = feature_config.dataset_dir + '-' + str(ratio)
        process(path_list=feature_config.class_names,
                raw_file_dir=os.path.join(dataset_dir, 'train'),
                raw_tar_dir=os.path.join(dataset_dir, 'train-' + str(ratio)),
                ratio_or_number=ratio,
                mode='copy',
                one=False)
        process(path_list=feature_config.class_names,
                raw_file_dir=os.path.join(dataset_dir, 'train'),
                raw_tar_dir=os.path.join(dataset_dir, 'train-' + str(ratio)),
                ratio_or_number=ratio,
                mode='copy',
                one=False)
        train_fun.train(save_dir=save_dir, dataset_dir=dataset_dir)





























