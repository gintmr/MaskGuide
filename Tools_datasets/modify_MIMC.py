#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将MIMC数据集的JSON文件转换为标准COCO格式
MIMC数据集与COCO的唯一差别在于：MIMC格式的annotations键下包含taxonomic和all两个键值对
转换后使用all键的内容作为annotations

annotations使用rle格式
"""
import json
import os
import glob


def convert_mimc_to_coco_format(input_path, output_path):
    """
    将单个MIMC格式JSON文件转换为标准COCO格式
    
    Args:
        input_path (str): 输入MIMC格式JSON文件路径
        output_path (str): 输出标准COCO格式JSON文件路径
    """
    print(f"Converting {input_path} to {output_path}")
    
    # 读取原始JSON文件
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查是否为MIMC格式
    if 'annotations' in data and isinstance(data['annotations'], dict):
        if 'all' in data['annotations']:
            # 提取all键的内容作为新的annotations
            new_annotations = data['annotations']['all']
            
            # 创建新的标准COCO格式数据
            new_data = {
                'images': data.get('images', []),
                'categories': data.get('categories', []),
                'annotations': new_annotations
            }
            
            # 保存转换后的文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, ensure_ascii=False, indent=2)
            
            print(f"Successfully converted {input_path} to standard COCO format")
            print(f"Images: {len(new_data['images'])}, Categories: {len(new_data['categories'])}, Annotations: {len(new_data['annotations'])}")
        else:
            print(f"Error: 'all' key not found in annotations of {input_path}")
            return False
    else:
        print(f"Warning: {input_path} is already in standard COCO format or has unexpected structure")
        # 如果已经是标准格式，直接复制
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    return True


def batch_convert_mimc_to_coco(input_dir, output_dir):
    """
    批量转换MIMC格式JSON文件到标准COCO格式
    
    Args:
        input_dir (str): 输入目录路径
        output_dir (str): 输出目录路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有JSON文件
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to convert")
    
    for json_file in json_files:
        filename = os.path.basename(json_file)
        output_path = os.path.join(output_dir, filename)
        
        success = convert_mimc_to_coco_format(json_file, output_path)
        if not success:
            print(f"Failed to convert {json_file}")


def main():
    # 默认路径设置
    input_dir = "/data2/wuxinrui/HF_data/MIMC_FINAL"  # MIMC数据集输入目录
    output_dir = "/data2/wuxinrui/HF_data/MIMC_FINAL_standard"  # 转换后的输出目录
    
    print("MIMC to COCO Format Converter")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    batch_convert_mimc_to_coco(input_dir, output_dir)
    
    print("\nConversion completed!")


if __name__ == "__main__":
    main()