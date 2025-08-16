from Tools_accelerate.visualize_feature import visualize_feature_map
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

def draw_contrast_feature_map(feature_map_list, chosen_mode_type, save_path):
    assert len(feature_map_list) == len(chosen_mode_type), "The length of feature_map_list and chosen_mode_type must be the same."
    
    length = len(feature_map_list)
    num_columns = 3  # 每行显示3张图片
    num_rows = (length + num_columns - 1) // num_columns  # 计算需要多少行

    width = num_columns * 5  # 每列宽度为4英寸
    height = num_rows * 5  # 每行高度为4英寸

    fig = plt.figure(figsize=(width, height))
    
    for i in range(length):
        feature_map = feature_map_list[i]
        mode_type = chosen_mode_type[i]
        row = i // num_columns
        col = i % num_columns
        ax = fig.add_subplot(num_rows, num_columns, i + 1)  # 添加子图
        ax.imshow(feature_map, cmap='viridis')  # 使用默认的色彩映射
        ax.set_title(mode_type)
        ax.axis('off')

    # 调整子图之间的间距
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close(fig)  # 关闭画布以释放资源


if __name__ == '__main__':
    chosen_mode_type = ['vit_t_vit_t_unmask', 'vit_t_vit_t_mask&unmask', 'vit_t_vit_t_mask','vit_t_vit_t_ori']
    
    features_folder = [os.path.join('/data2/wuxinrui/RA-L/MobileSAM/Tools_accelerate', mode_type) for mode_type in chosen_mode_type]
    
    chosen_mode_type.append('ori_img')
    
    features_name_lists = []
    for folder in features_folder:
        features_name_list = os.listdir(folder)
        features_name_list.sort()
        features_name_lists.append(set(features_name_list))

    common_features_names = set.intersection(*features_name_lists)

    for common_features_name in common_features_names:
        img_name = common_features_name.split('.npy')[0]
        feature_map_list = []
        for folder in features_folder:
            feature_map = np.load(os.path.join(folder, common_features_name))
            feature_map = visualize_feature_map(feature_map, method='pca')
            feature_map_list.append(feature_map)
        
        save_path = os.path.join('/data2/wuxinrui/RA-L/MobileSAM/Tools_accelerate/contrast_visualize', img_name)
        
        possible_img_folder = ['/data2/wuxinrui/RA-L/MobileSAM/NEW_MIMC_1024/images/train', '/data2/wuxinrui/RA-L/MobileSAM/NEW_MIMC_1024/images/val', '/data2/wuxinrui/Datasets/UIIS/UDW/train', '/data2/wuxinrui/Datasets/UIIS/UDW/val', '/data2/wuxinrui/Datasets/IMC_1000/MIMC_FINAL/seen/train_list', '/data2/wuxinrui/Datasets/IMC_1000/MIMC_FINAL/seen/test_list']
        for img_folder in possible_img_folder:
            ori_img_path = os.path.join(img_folder, img_name)
            if os.path.exists(ori_img_path):
                break
        img = cv2.imread(ori_img_path)
        feature_map_list.append(img)
        draw_contrast_feature_map(feature_map_list, chosen_mode_type, save_path)
        