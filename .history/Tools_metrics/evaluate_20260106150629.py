import os
import argparse
import random
import sys
from collections import defaultdict, deque
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from PIL import Image
import cv2
from eval_tools import calculate_segmentation_losses, calculate_metrics
import torch.nn as nn
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.distributed as dist
from torchvision import transforms
from mobile_sam.utils.transforms import ResizeLongestSide
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import segmentation_models_pytorch as smp
from transformers.models.maskformer.modeling_maskformer import dice_loss, sigmoid_focal_loss
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from eval_tools import init_model
from mobile_sam import sam_model_registry
from torch.cuda.amp import autocast


class Evaluate(nn.Module):
    mask_threshold: float = 0.5
    def __init__(
            self,
            model_type,
            checkpoint_path,
            batch_size=1,
            logger=None,
    ):
        super(Evaluate, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

        self.model_type = model_type
        self.model = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
        self.transform = ResizeLongestSide(self.model.image_encoder.img_size)
        self.model.to(device=self.device)
        self.batch_size = batch_size
        self.bbox_metric_dicts = {}
        self.point_metric_dicts = {}
        self.process_times = []
        self.logger = logger if logger is not None else None


    def forward(self, imgs, bboxes, labels, center_points, point_labels, original_input_size, resized_input_size, coco_image_names):
        import time
        start_time = time.time()
        imgs = imgs.to(self.device)
        
        #g 从输入的L,L图像 => （1024,1024）model.image_encoder.img_size
        input_images = torch.stack([self.model.preprocess(imgs[i,:,:,:]) for i in range(self.batch_size)], dim=0)
        input_images.to(self.device)
                
        try:
            # G input_images已完成缩放
            features = self.model.image_encoder(input_images)
            num_masks = sum([len(b) for b in bboxes])

            for feature, bbox, label, center_point, point_label, original_input_size_item, resized_input_size_item, coco_image_name, img in \
                    zip(features, bboxes, labels, center_points, point_labels, original_input_size, resized_input_size, coco_image_names, imgs):
                '''
                original_input_size_item: ori阶段图像尺寸 => H0,W0
                resized_input_size_item: middle阶段图像尺寸 => H1,W1(未填充成正方形)
                img: 填充成正方形(resized_input_size_item的长边)
                '''
                bbox = bbox.to(self.device)
                center_point = center_point.to(self.device)
                #G =================== point
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=(center_point, point_label),  #
                    boxes=None,
                    masks=None,
                )

                # Predict masks
                low_res_masks, iou_predictions = self.model.mask_decoder(
                    image_embeddings=feature.unsqueeze(0),
                    # image_embeddings=feature, #G 此处是predictor中的写法
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                # Upscale the masks to the original image resolution
                masks = F.interpolate(
                    low_res_masks,
                    (self.model.image_encoder.img_size, self.model.image_encoder.img_size),
                    mode="bilinear",
                    align_corners=False,
                ) ## self.model.image_encoder.img_size在vit_t设置下为1024
                
                resized_height, resized_width = img.shape[1], img.shape[2]
                masks = masks[..., : resized_height, : resized_width]
                # masks = F.interpolate(masks, original_input_size_item, mode="bilinear", align_corners=False)

                b,c,h,w=masks.shape

                label = label.unsqueeze(1)
                    #!------------设置metric-size，减少计算量------------
                # metric_size_item = (int(original_input_size_item[0]) // 2, int(original_input_size_item[1]) // 2)
                # label = F.interpolate(label.float(), metric_size_item, mode="bilinear", align_corners=False)
                # label = label.int()
                # masks = F.interpolate(masks, metric_size_item, mode="bilinear", align_corners=False)
                # assert masks.shape == label.shape, f"Masks shape {masks.shape} and label shape {label.shape} do not match"
                    #!------------设置metric-size，减少计算量------------
                pred_masks = masks.squeeze(1)
                iou_label = label.squeeze(1)
                # torch.cuda.empty_cache()
                eval_dict = calculate_metrics(pred_masks, iou_label)
                if self.point_metric_dicts == {}:
                    for metric_name, metric_list in eval_dict.items():
                        self.point_metric_dicts[metric_name] = []
                for metric_name, metric_list in eval_dict.items():
                    self.point_metric_dicts[metric_name].extend(metric_list[i] for i in range(len(metric_list)))
                #G =================== point

                #G =================== bbox
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=None,  #
                    boxes=bbox,
                    masks=None,
                )

                # Predict masks
                low_res_masks, iou_predictions = self.model.mask_decoder(
                    image_embeddings=feature.unsqueeze(0),
                    # image_embeddings=feature, #G 此处是predictor中的写法
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )                
                # Upscale the masks to the original image resolution
                masks = F.interpolate(
                    low_res_masks,
                    (self.model.image_encoder.img_size, self.model.image_encoder.img_size),
                    mode="bilinear",
                    align_corners=False,
                ) ## self.model.image_encoder.img_size在vit_t设置下为1024
                

                resized_height, resized_width = img.shape[1], img.shape[2]
                masks = masks[..., : resized_height, : resized_width]
                # masks = F.interpolate(masks, original_input_size_item, mode="bilinear", align_corners=False)


                b,c,h,w=masks.shape
                label = label.unsqueeze(1)
                    #!------------设置metric-size，减少计算量------------
                # metric_size_item = (int(original_input_size_item[0]) // 2, int(original_input_size_item[1]) // 2)
                # label = F.interpolate(label.float(), metric_size_item, mode="bilinear", align_corners=False)
                # label = label.int()
                # masks = F.interpolate(masks, metric_size_item, mode="bilinear", align_corners=False)
                # assert masks.shape == label.shape, f"Masks shape {masks.shape} and label shape {label.shape} do not match"
                    #!------------设置metric-size，减少计算量------------
                
                pred_masks = masks.squeeze(1)
                iou_label = label.squeeze(1)
                eval_dict = calculate_metrics(pred_masks, iou_label)
                if self.bbox_metric_dicts == {}:
                    for metric_name, metric_list in eval_dict.items():
                        self.bbox_metric_dicts[metric_name] = []
                for metric_name, metric_list in eval_dict.items():
                    self.bbox_metric_dicts[metric_name].extend(metric_list[i] for i in range(len(metric_list)))
                #G =================== point

        except Exception as e:
            # 获取图片名称
            img_names = [coco_image_name for coco_image_name in coco_image_names]
            print(f"Error occurred while processing images: {img_names}")
            print(f"Error details: {str(e)}")
            # 返回一个标记为无效的结果
            raise e
            # return {'valid': False}
            
        end_time = time.time()  # 记录结束时间
        self.process_times.append(end_time - start_time)  # 记录每个批次的处理时间
            
            
    def postprocess(self):
        bbox_avg_metrics = {}
        for metric_name, metric_list in self.bbox_metric_dicts.items():
            bbox_avg_metrics[metric_name] = np.mean(metric_list)
        for metric_name, avg_value in bbox_avg_metrics.items():
            print(f"bbox_{metric_name}: {avg_value}")
            # self.logger.info(f"bbox_{metric_name}: {avg_value}")
        
        point_avg_metrics = {}
        for metric_name, metric_list in self.point_metric_dicts.items():
            point_avg_metrics[metric_name] = np.mean(metric_list)
        for metric_name, avg_value in point_avg_metrics.items():
            print(f"point_{metric_name}: {avg_value}")
            # self.logger.info(f"point_{metric_name}: {avg_value}")
            
        avg_time_per_batch = np.mean(self.process_times)
        fps = self.batch_size / avg_time_per_batch
        print(f"fps: {fps}")
        # self.logger.info(f"Average FPS: {fps:.2f}")
        
