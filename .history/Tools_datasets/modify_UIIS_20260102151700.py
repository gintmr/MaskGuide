#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将UIIS数据集的JSON文件转换为标准COCO格式
UIIS数据集就是COCO格式，但是annotations中的segmentation 使用polygon格式，需要modify成rle格式
"""
import json
import os
from pycocotools import mask as maskUtils
from tqdm import tqdm


def polygon_to_rle(polygon, height, width):
    """
    将polygon格式的segmentation转换为RLE格式
    
    Args:
        polygon (list): polygon格式的segmentation，可以是单个多边形或多边形列表
        height (int): 图像高度
        width (int): 图像宽度
    
    Returns:
        dict: RLE格式的segmentation，包含'size'和'counts'键
    """
    # 如果polygon是空列表，返回None
    if not polygon or len(polygon) == 0:
        return None
    
    # 使用pycocotools将polygon转换为RLE
    # frPyObjects可以将polygon转换为RLE格式
    rle = maskUtils.frPyObjects(polygon, height, width)
    
    # 如果有多个多边形，需要合并它们
    if len(rle) > 1:
        rle = maskUtils.merge(rle)
    else:
        rle = rle[0]
    
    # 将counts从bytes转换为string（COCO标准格式）
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('utf-8')
    
    return rle


def convert_uiis_to_coco_format(input_path, output_path):
    """
    将UIIS格式JSON文件转换为标准COCO格式（polygon转RLE）
    
    Args:
        input_path (str): 输入UIIS格式JSON文件路径
        output_path (str): 输出标准COCO格式JSON文件路径
    """
    print(f"Converting {input_path} to {output_path}")
    
    # 读取原始JSON文件
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建图像ID到图像信息的映射，方便查找图像尺寸
    image_dict = {img['id']: img for img in data.get('images', [])}
    
    # 转换annotations中的segmentation格式
    new_annotations = []
    for ann in tqdm(data.get('annotations', []), desc="Converting annotations"):
        new_ann = ann.copy()
        
        # 获取对应的图像信息以获取尺寸
        image_id = ann.get('image_id')
        if image_id in image_dict:
            image_info = image_dict[image_id]
            height = image_info['height']
            width = image_info['width']
        else:
            print(f"Warning: Image ID {image_id} not found in images, skipping annotation {ann.get('id')}")
            continue
        
        # 转换segmentation格式
        if 'segmentation' in ann and ann['segmentation']:
            segmentation = ann['segmentation']
            
            # 检查是否为polygon格式（列表格式）
            if isinstance(segmentation, list) and len(segmentation) > 0:
                # 检查是否为polygon格式（第一个元素是列表）
                if isinstance(segmentation[0], list):
                    try:
                        rle = polygon_to_rle(segmentation, height, width)
                        if rle:
                            new_ann['segmentation'] = rle
                        else:
                            print(f"Warning: Failed to convert segmentation for annotation {ann.get('id')}")
                            continue
                    except Exception as e:
                        print(f"Error converting segmentation for annotation {ann.get('id')}: {e}")
                        continue
                # 如果已经是RLE格式（字典格式），保持不变
                elif isinstance(segmentation, dict):
                    new_ann['segmentation'] = segmentation
                else:
                    print(f"Warning: Unknown segmentation format for annotation {ann.get('id')}")
                    continue
            # 如果segmentation是空列表或None，跳过
            elif not segmentation:
                print(f"Warning: Empty segmentation for annotation {ann.get('id')}")
                continue
            else:
                print(f"Warning: Unknown segmentation format for annotation {ann.get('id')}")
                continue
        
        new_annotations.append(new_ann)
    
    # 创建新的标准COCO格式数据
    new_data = {
        'images': data.get('images', []),
        'categories': data.get('categories', []),
        'annotations': new_annotations
    }
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存转换后的文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully converted {input_path} to standard COCO format")
    print(f"Images: {len(new_data['images'])}, Categories: {len(new_data['categories'])}, Annotations: {len(new_data['annotations'])}")


def main():
    # 输入文件路径
    train_input = "/data2/wuxinrui/HF_data/COCO/annotations/instances_train2017.json"
    val_input = "/data2/wuxinrui/HF_data/COCO/annotations/instances_val2017.json"
    
    # 输出目录
    output_dir = "/data2/wuxinrui/HF_data/COCO_standard"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 输出文件路径
    train_output = os.path.join(output_dir, "train.json")
    val_output = os.path.join(output_dir, "val.json")
    
    print("UIIS to COCO Format Converter (Polygon to RLE)")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    # 转换train.json
    if os.path.exists(train_input):
        convert_uiis_to_coco_format(train_input, train_output)
        print("-" * 60)
    else:
        print(f"Warning: {train_input} not found, skipping...")
        print("-" * 60)
    
    # 转换val.json
    if os.path.exists(val_input):
        convert_uiis_to_coco_format(val_input, val_output)
        print("-" * 60)
    else:
        print(f"Warning: {val_input} not found, skipping...")
        print("-" * 60)
    
    print("\nConversion completed!")


if __name__ == "__main__":
    main()
