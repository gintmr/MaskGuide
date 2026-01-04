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
            add_distill=True,
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
        self.step_in_epoch = max_steps // epochs
        self.only_distill = only_distill  # 是否只蒸馏
        self.add_distill = add_distill
        self.distill_type = os.getenv("distill", "ori")
        self.features_folder_name = self.T_model_type + "_" + self.S_model_type + "_" + self.distill_type
        
        # 尝试使用 torch.compile 加速关键模块（PyTorch 2.0+）
        self._use_compile = hasattr(torch, 'compile') and os.getenv('USE_TORCH_COMPILE', '1') == '1'
        if self._use_compile:
            try:
                # 编译 mask decoder 和 prompt encoder（如果可能）
                if hasattr(self.S_model, 'mask_decoder'):
                    self.S_model.mask_decoder = torch.compile(self.S_model.mask_decoder, mode='reduce-overhead')
                if hasattr(self.S_model, 'prompt_encoder'):
                    self.S_model.prompt_encoder = torch.compile(self.S_model.prompt_encoder, mode='reduce-overhead')
            except Exception as e:
                print(f"Warning: torch.compile failed, falling back to normal mode: {e}")
                self._use_compile = False


    def forward(self, imgs, bboxes, labels, center_points, point_labels, original_input_size, resized_input_size, coco_image_names, validate=False):
        device = imgs.device
        imgs.to("cpu")
        assert self.T_model.image_encoder.img_size == self.S_model.image_encoder.img_size
        # imgs = random_resize(imgs)
        actual_batch_size = imgs.size(0)

        # print(f"imgs.shape = {imgs.shape}")

        if os.getenv("distill", "ori") == "ori":
            S_imgs = imgs
            T_imgs = imgs
            S_input_images = torch.stack([self.T_model.preprocess(S_imgs[i,:,:,:]) for i in range(actual_batch_size)], dim=0) #G padding
            T_input_images = torch.stack([self.T_model.preprocess(T_imgs[i,:,:,:]) for i in range(actual_batch_size)], dim=0) #G padding
            S_input_images.to(device)
            T_input_images.to(device)
        elif os.getenv("distill", "ori") == "mask" or os.getenv("distill", "ori") == "outline" or os.getenv("distill", "ori") == "unmask":
            #g 此时得到的imgs是两种格式数据沿通道纬度拼接起来的数据，其中前3个通道是正常img，后3个通道仅保留mask处像素的img
            # print("mask-distill mode!")
            S_imgs = imgs[:, :3, :, :]
            T_imgs = imgs[:, 3:, :, :]
            S_input_images = torch.stack([self.T_model.preprocess(S_imgs[i,:,:,:]) for i in range(actual_batch_size)], dim=0) #G padding
            T_input_images = torch.stack([self.T_model.preprocess(T_imgs[i,:,:,:]) for i in range(actual_batch_size)], dim=0) #G padding
            S_input_images.to(device)
            T_input_images.to(device)
            # print(f"S_imgs: {S_imgs.shape}, T_imgs: {T_imgs.shape}")
        elif os.getenv("distill", "ori") == "mask&outline":
            # import pdb 
            # pdb.set_trace()
            S_imgs = imgs[:, :3, :, :]
            T_imgs = S_imgs #! 仅用于后续可能的可视化，所以直接使用S_imgs
            T_imgs_mask = imgs[:, 3:6, :, :]
            T_imgs_outline = imgs[:, 6:, :, :]
            # print(f"S_imgs: {S_imgs.shape}, T_imgs: {T_imgs.shape}")
            S_input_images = torch.stack([self.T_model.preprocess(S_imgs[i,:,:,:]) for i in range(actual_batch_size)], dim=0) #G padding
            T_input_images_mask = torch.stack([self.T_model.preprocess(T_imgs_mask[i,:,:,:]) for i in range(actual_batch_size)], dim=0) #G padding
            T_input_images_outline = torch.stack([self.T_model.preprocess(T_imgs_outline[i,:,:,:]) for i in range(actual_batch_size)], dim=0) #G padding
            S_input_images.to(device)
            T_input_images_mask.to(device)
            T_input_images_outline.to(device)
            T_input_images = [T_input_images_mask, T_input_images_outline]
        elif "mask&unmask" in os.getenv("distill", "ori"):
            if validate:
                S_imgs = imgs[:, :3, :, :]
                T_imgs = S_imgs
                S_imgs_mask = imgs[:, 3:6, :, :]
                S_imgs_unmask = imgs[:, 6:, :, :]
                S_input_images = torch.stack([self.T_model.preprocess(S_imgs[i,:,:,:]) for i in range(actual_batch_size)], dim=0) #G padding
                S_input_images_mask = torch.stack([self.T_model.preprocess(S_imgs_mask[i,:,:,:]) for i in range(actual_batch_size)], dim=0) #G padding
                S_input_images_unmask = torch.stack([self.T_model.preprocess(S_imgs_unmask[i,:,:,:]) for i in range(actual_batch_size)], dim=0) #G padding
                S_input_images.to(device)
                S_input_images_mask.to(device)
                S_input_images_unmask.to(device)
                S_input_images = S_input_images
                
                validate_mode = os.getenv("validate", "v0")
                print(f"validate_mode = {validate_mode}!!!!!!!!!!!")
                if validate_mode == "v1":
                    S_input_images = [S_input_images_mask, S_input_images]
                elif validate_mode == "v2":
                    S_input_images = [S_input_images_mask, S_input_images_mask, S_input_images]
                elif validate_mode == "v3":
                    S_input_images = [S_input_images_mask, S_input_images, S_input_images]
                elif validate_mode == "v4":
                    S_input_images = [S_input_images_mask, S_input_images_mask, S_input_images_unmask]
                elif validate_mode == "v5":
                    S_input_images = [S_input_images_mask, S_input_images_unmask, S_input_images_unmask]
                elif validate_mode == "v6":
                    S_input_images = [S_input_images_mask, S_input_images_unmask]
                elif validate_mode == "v7":
                    S_input_images = [S_input_images_mask, S_input_images_unmask, S_input_images]
                # S_input_images = S_input_images_unmask
                # S_input_images = S_input_images_mask
                
            else:
                S_imgs = imgs[:, :3, :, :]
                T_imgs = S_imgs
                T_imgs_mask = imgs[:, 3:6, :, :]
                T_imgs_unmask = imgs[:, 6:, :, :]
                S_input_images = torch.stack([self.T_model.preprocess(S_imgs[i,:,:,:]) for i in range(actual_batch_size)], dim=0) #G padding
                T_input_images_mask = torch.stack([self.T_model.preprocess(T_imgs_mask[i,:,:,:]) for i in range(actual_batch_size)], dim=0) #G padding
                T_input_images_unmask = torch.stack([self.T_model.preprocess(T_imgs_unmask[i,:,:,:]) for i in range(actual_batch_size)], dim=0) #G padding
                S_input_images.to(device)
                T_input_images_mask.to(device)
                T_input_images_unmask.to(device)
                if os.getenv("distill", "ori") == "mask&unmask":
                    T_input_images = [T_input_images_mask, S_input_images]
                elif os.getenv("distill", "ori") == "mask&unmask_neg":
                    T_input_images = [T_input_images_mask, T_input_images_unmask]
                elif os.getenv("distill", "ori") == "mask&unmask_v1":
                    T_input_images = [T_input_images_mask, T_input_images_unmask, S_input_images]
                # S_input_images = S_input_images_unmask
                # S_input_images = S_input_images_mask
        try:
            RATE = self.global_step / self.max_steps
            os.environ['RATE'] = str(RATE)
            if isinstance(S_input_images, list):
                S_features_list = [self.S_model.image_encoder(S_input_images[i]) for i in range(len(S_input_images))]
                S_features = sum(S_features_list) / len(S_features_list)
            else:            
                S_features = self.S_model.image_encoder(S_input_images)

            if not validate and self.add_distill:
                distill_loss = torch.tensor(0, device="cuda", dtype=torch.float32)

                example_feature_name = coco_image_names[0] + ".npy"
                if isinstance(T_input_images, list):
                    T_features_list = [self.T_model.image_encoder(T_input_images[i]) for i in range(len(T_input_images))]
                    T_features = sum(T_features_list) / len(T_features_list)
                else:
                    T_features = self.T_model.image_encoder(T_input_images)

                if isinstance(T_features, list):
                    print("T_features is a list")
                    for T_feature in T_features:
                        device = T_feature.device
                        distill_loss = distill_loss.to(device)
                        distill_loss += self.feature_distillation_loss(T_feature, S_features, RATE)
                else:
                    distill_loss = self.feature_distillation_loss(T_features, S_features, RATE)
                # distill_loss = self.new_feature_distillation_loss(T_features, S_features, RATE)
            else:
                distill_loss = torch.tensor(0)

            num_masks = sum([len(b) for b in bboxes])

            BS_LOSS = 0
            BS_loss_IoU, BS_loss_dice, BS_loss_tversky, BS_loss_mse, BS_loss_focal = 0, 0, 0, 0, 0
            BS_IoU, BS_dice = 0, 0
            
            if self.multimask:
                num_masks *= 3

            Finetune_dict = None
            if self.only_distill and not validate:
                pass
            else:
                for S_feature, bbox, label, center_point, point_label, original_input_size_item, resized_input_size_item, coco_image_name, img, T_img in \
                        zip(S_features, bboxes, labels, center_points, point_labels, original_input_size, resized_input_size, coco_image_names, S_imgs, T_imgs):
                    '''
                    original_input_size_item: ori阶段图像尺寸 => H0,W0
                    resized_input_size_item: middle阶段图像尺寸 => H1,W1(未填充成正方形)
                    img: 填充成正方形(resized_input_size_item的长边)
                    '''
                    Finetune_dict = finetune(feature=S_feature, bbox=bbox, label=label, center_point=center_point, point_label=point_label, original_input_size_item=original_input_size_item, resized_input_size_item=resized_input_size_item, coco_image_name=coco_image_name, img=img, T_img=T_img,  training_visual_path=f"/data2/wuxinrui/Distill-SAM/training_visual_distill/{self.current_epoch}", self=self, validate=validate)

                    #g 排除低质样本的副作用
                    BS_LOSS += Finetune_dict["total_loss"]
                    BS_loss_IoU += Finetune_dict["iou_loss"]
                    # BS_loss_dice += Finetune_dict["dice_loss"]
                    # BS_loss_tversky += Finetune_dict["tversky_loss"]
                    BS_loss_mse += Finetune_dict["mse_loss"]
                    BS_loss_focal += Finetune_dict["focal_loss"]
                    BS_IoU += Finetune_dict["iou"]
                    BS_dice += Finetune_dict["dice"]

            # BS_LOSS = BS_LOSS.clone().detach().to(device).requires_grad_(True)

            BS = len(imgs)
            av_BS_IoU = BS_IoU / BS
            av_BS_dice = BS_dice / BS
            av_BS_loss_IoU = BS_loss_IoU / BS
            # av_BS_loss_dice = BS_loss_dice / BS
            # av_BS_loss_tversky = BS_loss_tversky / BS
            av_BS_loss_mse = BS_loss_mse / BS
            av_BS_loss_focal = BS_loss_focal / BS
            if BS_LOSS == 0:
                pass
            else:
                assert not torch.isnan(BS_LOSS), "loss is nan"
            if Finetune_dict:
                iou =  Finetune_dict["iou"]
            else:
                iou = 0

            if iou < 0.9:
                loss = BS_LOSS + distill_loss * self.distill_weight
            else:
                loss = BS_LOSS + (distill_loss * self.distill_weight) / 2

            assert not torch.isnan(loss), "loss is nan"

            return {
                "RATE": RATE,
                'av_BS_IoU': av_BS_IoU,
                'av_BS_dice': av_BS_dice,
                'loss': loss,
                # 'av_BS_loss_tversky': av_BS_loss_tversky,
                # 'av_BS_loss_dice': av_BS_loss_dice,
                'av_BS_loss_IoU': av_BS_loss_IoU,
                'av_BS_loss_mse': av_BS_loss_mse,
                'av_BS_loss_focal': av_BS_loss_focal,
                'distill_loss': distill_loss,
            }

        except Exception as e:
            # 获取图片名称
            img_names = [coco_image_name for coco_image_name in coco_image_names]
            print(f"Error occurred while processing images: {img_names}")
            print(f"Error details: {str(e)}")
            raise e

    def training_step(self, batch, batch_nb):
        imgs, bboxes, labels, center_points, point_labels, img_name ,original_input_size,resized_input_size, coco_image_names = batch

        # with autocast():
        #     outputs = self(imgs, bboxes, labels, center_points, point_labels,category_ids,original_input_size,resized_input_size, coco_image_names)

        outputs = self(imgs, bboxes, labels, center_points, point_labels ,original_input_size,resized_input_size, coco_image_names)

        if outputs['loss'] == 0:
            print("Skipping invalid batch")
            return None

        metrics = outputs
        log_metrics = {k: v for k, v in outputs.items() if k != 'loss'}
        self.log_dict(log_metrics, prog_bar=True, rank_zero_only=True)
        del outputs, log_metrics
        torch.cuda.empty_cache()
        return metrics


    def validation_step(self, batch, batch_idx):
        imgs, bboxes, labels, center_points, point_labels, img_name ,original_input_size, resized_input_size, coco_image_names = batch

        # with autocast():
        #     outputs = self(imgs, bboxes, labels, center_points, point_labels,category_ids,original_input_size,resized_input_size, coco_image_names)
        with torch.no_grad():
            outputs = self(imgs, bboxes, labels, center_points, point_labels,original_input_size,resized_input_size, coco_image_names, validate=True)

        if outputs['loss'] == 0:
            print("Skipping invalid batch")
            return None
        metrics = {}
        metrics['av_BS_IoU'] = outputs['av_BS_IoU']
        metrics['loss'] = outputs['loss']
        return metrics
        
            
    def validation_epoch_end(self, outputs):
        if not outputs:
            print("No valid batches in validation step.")
            return

        # 过滤掉None值
        valid_outputs = [output for output in outputs if output is not None]
        if not valid_outputs:
            print("No valid outputs in validation step.")
            return

        # 汇总所有批次的指标
        avg_metrics = {}
        for key in valid_outputs[0].keys():
            values = [output[key] for output in valid_outputs if key in output]
            if values:
                # 处理tensor类型，转换为float
                if isinstance(values[0], torch.Tensor):
                    values = [v.item() if v.numel() == 1 else float(v) for v in values]
                avg_metrics[key] = sum(values) / len(values)  # 直接对 float 类型的指标进行平均

        # 记录汇总后的指标到日志中
        for key, value in avg_metrics.items():
            self.log(f'val_{key}', value, on_epoch=True, sync_dist=True)