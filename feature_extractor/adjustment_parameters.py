from feature_extractor import train_fun, feature_config

if __name__ == '__main__':
    lr_list = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    for i, lr in enumerate(lr_list):
        save_dir = 'lr-' + feature_config.save_dir + '-' + str(i)
        train_fun.train(save_dir=save_dir, learning_rate=lr)
















