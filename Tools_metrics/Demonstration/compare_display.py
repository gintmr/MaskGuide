import sys
sys.path.append('/data2/wuxinrui/RA-L/MobileSAM')

from Datasets.coco import Coco2MaskDataset
import torch
from eval_tools import overlay_mask_on_image, combine_visualize_results
import cv2
import numpy as np
IMG_path = "/data2/wuxinrui/RA-L/MobileSAM/NEW_MIMC_1024/images/train"
GT_path = "/data2/wuxinrui/RA-L/MobileSAM/NEW_MIMC_1024/annotations/train.json"

datasets = Coco2MaskDataset(data_root=IMG_path, annotation_path=GT_path, image_size=1024,length=200,num_points=10,use_centerpoint=False)

dataloader = torch.utils.data.DataLoader(datasets, batch_size=1, shuffle=True, num_workers=4)

for i, data in enumerate(dataloader):
    imgs, bboxes, labels, center_points, point_labels, img_names, target_labels, original_input_size, resized_input_size, coco_image_names = data
    
    for img, label, coco_image_name in zip(imgs,labels,coco_image_names):
        try:
            label = label.squeeze()
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            if isinstance(label, torch.Tensor):
                label = label.cpu().numpy()
                
            label = label.astype(np.uint8)
            GT_array = overlay_mask_on_image(label, image_array=img, array_out=True, save_img=False)
            IMG_array = img.transpose((1, 2, 0))
            original_image_with_boundaries = IMG_array.copy()
            contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(original_image_with_boundaries, contours, -1, (255, 255, 255), 5)
            Mask_stroke_array = original_image_with_boundaries

            output_path_combined = f"/data2/wuxinrui/RA-L/MobileSAM/Tools_metrics/Demonstration/display/{coco_image_name}"
            combine_visualize_results(GT_array, IMG_array, Mask_stroke_array, output_path_combined)

            
        except:
            print("Error processing image:", coco_image_name)