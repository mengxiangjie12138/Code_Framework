# main
source_dir_list = [r'D:\My_Code\mengxiangjie\My_Code_Framework\evaluation\data_and_model\NCGCN']
# all_evaluation = ['confusion_matrix', 'evaluation_value', 'ROC', 't_sne', 'accuracy_of_each_class']
evaluation_fun_list = ['evaluation_value', 'accuracy_of_each_class']

labels_true_name = 'labels_true.npy'
labels_pred_name = 'labels_pred.npy'
present_name = 'present.npy'
features_name = 'features.npy'

class_names = ['gametocyte', 'leukocyte', 'red_blood_cell', 'ring', 'schizont', 'trophozoite']
# class_names = ['Parasitized', 'Uninfected']
# class_names = ['Babesia', 'LiShiMan', 'Malaria', 'maodichong', 'RB Cell', 'Toxoplasma', 'White Cell', 'ZhuiXingChong']
# class_names = ['Babesia', 'Malaria', 'RB Cell', 'White Cell']

class_number = len(class_names)

# ROC
each_class_plot = True

# t_sne
tsne_balance = False
tsne_point_num = 100











