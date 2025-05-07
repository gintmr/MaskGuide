import torch
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
import os 
import random
from eval_tools import calculate_pa_iou, overlay_mask_on_image,overlay_point_on_image, combine_visualize_results, calculate_segmentation_losses
import random
from Tools_finetune.loss_calculate import OceanSegmentationLoss

oceanegmentationLoss = OceanSegmentationLoss()

def finetune(**kwargs):
    feature, bbox, label, center_point, point_label, target_label, original_input_size_item, resized_input_size_item, coco_image_name, img = kwargs['feature'], kwargs['bbox'], kwargs['label'], kwargs['center_point'], kwargs['point_label'], kwargs['target_label'], kwargs['original_input_size_item'], kwargs['resized_input_size_item'], kwargs['coco_image_name'], kwargs['img']

    training_visual_path = kwargs['training_visual_path']

    self = kwargs['self']
    if hasattr(self, 'model'):
        model = self.model
    elif hasattr(self, 'S_model'):
        model = self.T_model
    
    if random.random() < 0.8:
        self.use_bbox  = False
    else:
        self.use_bbox  = True
    if self.use_bbox:
        bbox_input=bbox
    else:
        bbox_input=None


    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=(center_point, point_label),
        boxes=bbox_input,
        masks=None,
    )
    torch.cuda.empty_cache()


    # Predict masks
    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings=feature.unsqueeze(0),
        # image_embeddings=feature, #G 此处是predictor中的写法
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=self.multimask,
    )
    torch.cuda.empty_cache()
    
    # Upscale the masks to the original image resolution
    masks = F.interpolate(
        low_res_masks,
        (model.image_encoder.img_size, model.image_encoder.img_size),
        mode="bilinear",
        align_corners=False,
    ) ## model.image_encoder.img_size在vit_t设置下为1024
    torch.cuda.empty_cache()
    del low_res_masks
    

    resized_height, resized_width = img.shape[1], img.shape[2]
    masks = masks[..., : resized_height, : resized_width]
    # masks = F.interpolate(masks, original_input_size_item, mode="bilinear", align_corners=False)
    torch.cuda.empty_cache()

    b,c,h,w=masks.shape
    if self.multimask:
        label = torch.repeat_interleave(label.unsqueeze(1), masks.shape[1], dim=1).view(b*3,-1,h,w)
        masks = masks.view(b*3,-1,h,w)
        iou_predictions = iou_predictions.reshape(-1,1)
    else:
        label = label.unsqueeze(1)
        #!------------设置metric-size，减少计算量------------
    # metric_size_item = (int(original_input_size_item[0]) // 2, int(original_input_size_item[1]) // 2)
    # label = F.interpolate(label.float(), metric_size_item, mode="bilinear", align_corners=False)
    # label = label.int()
    # masks = F.interpolate(masks, metric_size_item, mode="bilinear", align_corners=False)
    # assert masks.shape == label.shape, f"Masks shape {masks.shape} and label shape {label.shape} do not match"
        #!------------设置metric-size，减少计算量------------
    
    #G ------- 计算IoU + 可视化输出--------G#
    pred_masks = masks.squeeze(1)
    iou_label = label.squeeze(1)
    torch.cuda.empty_cache()
    
    loss_dict = oceanegmentationLoss(pred_masks, iou_label)
    
    ext_name = coco_image_name.split('.')[-1]
    if not os.path.exists(training_visual_path):
        os.makedirs(training_visual_path, exist_ok=True)
    #G 1% to save images
    # if random.random() < 0.03:
    output_path_combined = os.path.join(training_visual_path, f"{self.current_epoch}_{coco_image_name.replace(ext_name, '_combined.jpg')}")

    pred_mask_array = overlay_mask_on_image(pred_masks, image_array=img, array_out=True, save_img=False)
    GT_mask_array = overlay_mask_on_image(iou_label, image_array=img, array_out=True, save_img=False)
    point_array = overlay_point_on_image(center_point, image_array=img, array_out=True, save_img=False)
    combine_visualize_results(pred_mask_array, GT_mask_array, point_array, output_path_combined)

    #G check to avoid too many images in the folder
    image_files = [f for f in os.listdir(training_visual_path) if f.endswith(('_combined.jpg'))]
    if len(image_files) > 100:
        #G randomly select three file prefixes
        random_prefixes = random.sample(set(f.split('_combined.jpg')[0] for f in image_files), 3)
        for prefix in random_prefixes:
            suffix = ('_combined.jpg')
            file_to_delete = os.path.join(training_visual_path, f"{prefix}{suffix}")
            if os.path.exists(file_to_delete):
                try: #G 防止误删可视化图后报错
                    os.remove(file_to_delete)
                except:
                    pass
    #G ------- 计算IoU + 可视化输出--------G#


    return loss_dict