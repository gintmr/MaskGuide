import cv2
import numpy as np
import os

def calculate_mean_and_std(image_folder):
    """计算文件夹中所有图像的平均均值和标准差"""
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    mean_list = []
    std_list = []
    
    for image_path in image_files:
        image = cv2.imread(image_path).astype(np.float32) / 255.0
        mean = np.mean(image, axis=(0, 1))
        std = np.std(image, axis=(0, 1))
        mean_list.append(mean)
        std_list.append(std)
    
    mean_array = np.array(mean_list)
    std_array = np.array(std_list)
    
    overall_mean = np.mean(mean_array, axis=0)
    overall_std = np.mean(std_array, axis=0)
    
    return overall_mean, overall_std

def color_transfer(source, target_mean, target_std):
    """将水下参考图像的颜色风格应用到目标图像"""
    source = source.astype(np.float32) / 255.0
    
    # 计算源图像的均值和标准差
    source_mean = np.mean(source, axis=(0, 1))
    source_std = np.std(source, axis=(0, 1))
    
    # 颜色迁移公式
    img_normalized = (source - source_mean) / source_std
    img_transfer = img_normalized * target_std + target_mean
    
    img_transfer = np.clip(img_transfer, 0, 1) * 255.0
    return img_transfer.astype(np.uint8)

def process_folders(source_folder, target_folder, output_folder):
    """处理两个文件夹中的所有图像"""
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 计算所有海底图像的平均均值和标准差
    target_mean, target_std = calculate_mean_and_std(target_folder)
    
    # 获取文件夹中的所有图像文件
    source_files = sorted([os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.endswith(('.jpg', '.png'))])
    
    for source_path in source_files:
        # 读取图像
        source_image = cv2.imread(source_path)
        
        if source_image is None:
            print(f"无法读取图像：{source_path}")
            continue
        
        # 应用颜色迁移
        result_image = color_transfer(source_image, target_mean, target_std)
        
        # 保存结果
        output_filename = os.path.basename(source_path)
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, result_image)
        print(f"处理完成并保存到：{output_path}")

# 示例使用
source_folder = '/data2/wuxinrui/Datasets/COCO/images/train2017'  # 正常图像文件夹
target_folder = '/data2/wuxinrui/Distill-SAM/NEW_MIMC_1024/images/train'  # 水下风格参考图像文件夹
output_folder = '/data2/wuxinrui/Datasets/COCO/images/train2017_marine_style'  # 输出文件夹

process_folders(source_folder, target_folder, output_folder)