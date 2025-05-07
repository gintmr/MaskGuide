from .DistillFintuner import AbstractDistillFinetuner
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
import os 
import random
from eval_tools import calculate_pa_iou, overlay_mask_on_image,overlay_point_on_image, combine_visualize_results, calculate_segmentation_losses
from Tools_finetune.finetuner import finetune


class Imgencoder_Distill(AbstractDistillFinetuner):
    mask_threshold: float = 0.5
    def __init__(
            self,
            T_model,
            S_model,
            T_checkpoint_path,
            S_checkpoint_path,
            freeze_image_encoder=False,
            freeze_prompt_encoder=False,
            freeze_mask_decoder=False,
            batch_size=1,
            learning_rate=1e-5,
            weight_decay=1e-4, #G avoid overfitting
            train_dataset=None,
            val_dataset=None,
            metrics_interval=20,
            multimask=False,
            use_bbox=False,
            max_steps=10000,
            epochs=10,
            distill_weight=3,  # 新增参数：蒸馏权重
            only_distill=False,  # 新增参数：是否只蒸馏
    ):
        super(Imgencoder_Distill, self).__init__(
            T_model=T_model,
            S_model=S_model,
            T_checkpoint_path=T_checkpoint_path,
            S_checkpoint_path=S_checkpoint_path,
            freeze_image_encoder=freeze_image_encoder,
            freeze_prompt_encoder=freeze_prompt_encoder,
            freeze_mask_decoder=freeze_mask_decoder,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            metrics_interval=metrics_interval,
            multimask=multimask,
            use_bbox=use_bbox
        )

        self.distill_weight = distill_weight  # 蒸馏权重
        self.max_steps = max_steps
        self.current_step = 0
        self.step_in_epoch = max_steps // epochs
        self.only_distill = only_distill  # 是否只蒸馏
        
    def forward(self, imgs, bboxes, labels, center_points, point_labels, target_labels, original_input_size, resized_input_size, coco_image_names):
        device = imgs.device
        imgs.to("cpu")
        assert self.T_model.image_encoder.img_size == self.S_model.image_encoder.img_size
        # imgs = random_resize(imgs)

        input_images = torch.stack([self.T_model.preprocess(imgs[i,:,:,:]) for i in range(self.batch_size)], dim=0) #G padding
        input_images.to(device)
        try:
            T_features, T_layers_features = self.T_model.image_encoder(input_images)
            S_features, S_layers_features = self.S_model.image_encoder(input_images)

            RATE = self.current_step / self.max_steps
            distill_loss = self.feature_distillation_loss(T_features, S_features, T_layers_features, S_layers_features, RATE)

            del T_layers_features, S_layers_features
            num_masks = sum([len(b) for b in bboxes])

            BS_LOSS = 0
            BS_loss_IoU, BS_loss_dice, BS_loss_tversky, BS_loss_mse = 0, 0, 0, 0
            BS_IoU, BS_dice = 0, 0
            
            if self.multimask:
                num_masks *= 3

            if self.only_distill:
                pass
            else:
                for T_feature, S_feature, bbox, label, center_point, point_label, target_label, original_input_size_item, resized_input_size_item, coco_image_name, img in \
                        zip(T_features, S_features, bboxes, labels, center_points, point_labels, target_labels, original_input_size, resized_input_size, coco_image_names, imgs):
                    '''
                    original_input_size_item: ori阶段图像尺寸 => H0,W0
                    resized_input_size_item: middle阶段图像尺寸 => H1,W1(未填充成正方形)
                    img: 填充成正方形(resized_input_size_item的长边)
                    '''
                    Finetune_dict = finetune(feature=S_feature, bbox=bbox, label=label, center_point=center_point, point_label=point_label, target_label=target_label, original_input_size_item=original_input_size_item, resized_input_size_item=resized_input_size_item, coco_image_name=coco_image_name, img=img, training_visual_path=f"/data2/wuxinrui/RA-L/MobileSAM/training_visual_distill/{self.current_epoch}", self=self)

                    #g 排除低质样本的副作用
                    BS_LOSS += Finetune_dict["total_loss"]
                    BS_loss_IoU += Finetune_dict["iou_loss"]
                    BS_loss_dice += Finetune_dict["dice_loss"]
                    BS_loss_tversky += Finetune_dict["tversky_loss"]
                    BS_loss_mse += Finetune_dict["mse_loss"]
                    BS_IoU += Finetune_dict["iou"]
                    BS_dice += Finetune_dict["dice"]

            BS = len(imgs)
            av_BS_IoU = BS_IoU / BS
            av_BS_dice = BS_dice / BS
            av_BS_loss_IoU = BS_loss_IoU / BS
            av_BS_loss_dice = BS_loss_dice / BS
            av_BS_loss_tversky = BS_loss_tversky / BS
            av_BS_loss_mse = BS_loss_mse / BS

            loss = BS_LOSS + distill_loss * self.distill_weight

            assert not torch.isnan(loss), "loss is nan"
            return {
                "RATE": RATE,
                'av_BS_IoU': av_BS_IoU,
                'av_BS_dice': av_BS_dice,
                'loss': loss,
                'av_BS_loss_tversky': av_BS_loss_tversky,
                'av_BS_loss_dice': av_BS_loss_dice,
                'av_BS_loss_IoU': av_BS_loss_IoU,
                'av_BS_loss_mse': av_BS_loss_mse,
                'distill_loss': distill_loss,
            }

        except Exception as e:
            # 获取图片名称
            img_names = [coco_image_name for coco_image_name in coco_image_names]
            print(f"Error occurred while processing images: {img_names}")
            print(f"Error details: {str(e)}")
            raise e
        
    def training_step(self, batch, batch_nb):
        imgs, bboxes, labels, center_points, point_labels, img_name, category_ids ,original_input_size,resized_input_size, coco_image_names = batch

        with autocast():
            outputs = self(imgs, bboxes, labels, center_points, point_labels,category_ids,original_input_size,resized_input_size, coco_image_names)
            self.current_step += 1

        if outputs['loss'] == 0:
            print("Skipping invalid batch")
            return None
        
        metrics = outputs
        log_metrics = {k: v for k, v in outputs.items() if k != 'loss'}
        self.log_dict(log_metrics, prog_bar=True, rank_zero_only=True)
        del outputs, log_metrics
        torch.cuda.empty_cache()
        return metrics
