import torch
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
import os 
import random
from eval_tools import calculate_pa_iou, overlay_mask_on_image, overlay_point_on_image, combine_visualize_results, calculate_segmentation_losses, combine_visualize_results_only2
import random
from Tools_finetune.loss_calculate import OceanSegmentationLoss

oceanegmentationLoss = OceanSegmentationLoss()

# 尝试使用 torch.compile 加速（PyTorch 2.0+）
_USE_COMPILE = hasattr(torch, 'compile') and os.getenv('USE_TORCH_COMPILE', '1') == '1'

def finetune(**kwargs):
    feature, bbox, label, center_point, point_label, original_input_size_item, resized_input_size_item, coco_image_name, img, T_img = kwargs['feature'], kwargs['bbox'], kwargs['label'], kwargs['center_point'], kwargs['point_label'], kwargs['original_input_size_item'], kwargs['resized_input_size_item'], kwargs['coco_image_name'], kwargs['img'], kwargs['T_img']

    validate = kwargs['validate']

    training_visual_path = kwargs['training_visual_path']

    self = kwargs['self']
    if hasattr(self, 'model'):
        model = self.model
    elif hasattr(self, 'S_model'):
        model = self.S_model


    if validate:
        if os.getenv("test_prompts", "bbox") == "bbox":
            points = None
            self.use_bbox = True
        elif os.getenv("test_prompts", "bbox") == "point":
            points = (center_point, point_label)
            self.use_bbox = False


    else:
        if random.random() < 0.5:
            self.use_bbox  = False
        else:
            self.use_bbox  = True
        
        points = (center_point, point_label)

    if self.use_bbox:
        bbox_input=bbox
    else:
        bbox_input=None


    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=points,
        boxes=bbox_input,
        masks=None,
    )

    # Predict masks
    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings=feature.unsqueeze(0),
        # image_embeddings=feature, #G 此处是predictor中的写法
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=self.multimask,
    )
    
    # Upscale the masks to the original image resolution
    masks = F.interpolate(
        low_res_masks,
        (model.image_encoder.img_size, model.image_encoder.img_size),
        mode="bilinear",
        align_corners=False,
    ) ## model.image_encoder.img_size在vit_t设置下为1024
    del low_res_masks

    resized_height, resized_width = img.shape[1], img.shape[2]
    masks = masks[..., : resized_height, : resized_width]
    # masks = F.interpolate(masks, original_input_size_item, mode="bilinear", align_corners=False)

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
    #g 此处的pred_masks就是输出概率值。（范围+-N）
    #g 计算损失时，常会对齐使用sigmoid函数，将概率值限制在0-1之间 —— pred_probs (0-1)
    #g pred_masks的>0即对应pred_probs的>0.5


    ext_name = coco_image_name.split('.')[-1]
    if not os.path.exists(training_visual_path):
        os.makedirs(training_visual_path, exist_ok=True)
    #G 1% to save images
    # if random.random() < 0.03:

    if not validate:
        if random.random() < 0.05:
            output_path_combined = os.path.join(training_visual_path, f"{self.current_epoch}_{coco_image_name.replace(ext_name, '_combined.jpg')}")
            pred_mask_array = overlay_mask_on_image(pred_masks, image_array=img, array_out=True, save_img=False)
            GT_mask_array = overlay_mask_on_image(iou_label, image_array=img, array_out=True, save_img=False)
            point_array = overlay_point_on_image(center_point, image_array=img, array_out=True, save_img=False)
            combine_visualize_results(pred_mask_array, GT_mask_array, point_array, output_path_combined)
            #G check to avoid too many images in the folder
            image_files = [f for f in os.listdir(training_visual_path) if f.endswith(('_combined.jpg'))]
            if len(image_files) > 100:
                delta = len(image_files) - 100
                #G randomly select three file prefixes
                random_prefixes = random.sample(set(f.split('_combined.jpg')[0] for f in image_files), delta)
                for prefix in random_prefixes:
                    suffix = ('_combined.jpg')
                    file_to_delete = os.path.join(training_visual_path, f"{prefix}{suffix}")
                    if os.path.exists(file_to_delete):
                        try: #G 防止误删可视化图后报错
                            os.remove(file_to_delete)
                        except:
                            pass
    else:
        val_visual_pth = training_visual_path.replace("training_visual_distill", "valing_visual_distill")
        os.makedirs(val_visual_pth, exist_ok=True)
        
        if random.random() < 0.1:
            pred_mask_array = overlay_mask_on_image(pred_masks, image_array=img, array_out=True, save_img=False)
            GT_mask_array = overlay_mask_on_image(iou_label, image_array=img, array_out=True, save_img=False)
            output_path_combined = os.path.join(val_visual_pth, f"{self.current_epoch}_{coco_image_name.replace(ext_name, '_combined.jpg')}")
            ori_img = T_img.cpu().numpy().transpose(1, 2, 0)
            combine_visualize_results(pred_mask_array, GT_mask_array, ori_img, output_path_combined)
            # combine_visualize_results_only2(pred_mask_array, GT_mask_array, output_path_combined)
            #G check to avoid too many images in the folder
            image_files = [f for f in os.listdir(val_visual_pth) if f.endswith(('_combined.jpg'))]
            if len(image_files) > 100:
                delta = len(image_files) - 100
                #G randomly select three file prefixes
                random_prefixes = random.sample(set(f for f in image_files), delta)
                for prefix in random_prefixes:
                    file_to_delete = os.path.join(val_visual_pth, prefix)
                    if os.path.exists(file_to_delete):
                        try: #G 防止误删可视化图后报错
                            os.remove(file_to_delete)
                        except:
                            pass
    #G ------- 计算IoU + 可视化输出--------G#

    loss_dict = oceanegmentationLoss(pred_masks, iou_label)
    del pred_masks, iou_label
    return loss_dict