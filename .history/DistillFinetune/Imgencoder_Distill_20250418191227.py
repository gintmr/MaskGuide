from .DistillFintuner import AbstractDistillFinetuner
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
import os 
import random
from eval_tools import calculate_pa_iou, overlay_mask_on_image,overlay_point_on_image, combine_visualize_results, calculate_segmentation_losses

def feature_distillation_loss(T_features, S_features, T_layers_features, S_layers_features, RATE, reduction='mean', alpha=0.5, beta=0.1):
    """
    计算教师模型和学生模型特征的蒸馏损失。

    参数:
        T_features (torch.Tensor): 教师模型的特征输出。
        S_features (torch.Tensor): 学生模型的特征输出。
        T_layers_features (dict): 教师模型的各层特征输出。
        S_layers_features (dict): 学生模型的各层特征输出。
        reduction (str): 损失的归约方式，可选 'mean' 或 'sum'。
        alpha (float): MSE 损失的权重。
        beta (float): 余弦相似度损失的权重。

    返回:
        torch.Tensor: 蒸馏损失。
    """
    total_loss = 0.0
    # 计算 MSE 损失
    mse_loss = F.mse_loss(S_features, T_features.detach(), reduction=reduction)

    # 计算余弦相似度损失
    T_features_norm = F.normalize(T_features.detach(), p=2, dim=1)
    S_features_norm = F.normalize(S_features, p=2, dim=1)
    cosine_similarity = F.cosine_similarity(S_features_norm, T_features_norm, dim=1)
    cosine_loss = 1 - cosine_similarity.mean()

    total_loss = alpha * mse_loss + beta * cosine_loss
    
    
    temperature = max(3.0 - 2*(RATE), 1.0)
    
    # 阶段感知权重调整
    progress = RATE
    if progress < 0.5:  # 前期侧重特征对齐
        A,B = 0.7, 0.1
    else:  # 后期加强注意力对齐
        A,B = 0.4, 0.1
    
    for T_layers_feature, S_layers_feature in zip(T_layers_features.values(), S_layers_features.values()):
        # 计算 MSE 损失
        if S_layers_feature.shape[-1:] != T_layers_feature.shape[-1:]:
            #G interporlate需要输入四维张量，这里将三维张量扩充为四维张量
            S_layers_feature = S_layers_feature.unsqueeze(-1)  # 形状变为 [1, 4096, 160, 1]
            T_layers_feature = T_layers_feature.unsqueeze(-1)  # 形状变为 [1, 4096, 128, 1]

            # 使用 F.interpolate 进行插值
            S_layers_feature = F.interpolate(S_layers_feature, size=T_layers_feature.shape[-2:], mode='bilinear', align_corners=False)

            # 将四维张量还原为三维
            S_layers_feature = S_layers_feature.squeeze(-1)  
            T_layers_feature = T_layers_feature.squeeze(-1)  
            
        layers_mse_loss = F.mse_loss(S_layers_feature/temperature, T_layers_feature.detach()/temperature, reduction=reduction) * (temperature ** 2)
        layers_cosine_loss = 1 - F.cosine_similarity(F.normalize(S_layers_feature, p=2, dim=1), F.normalize(T_layers_feature.detach(), p=2, dim=1), dim=1).mean()
        
        total_loss += A*layers_mse_loss + B*layers_cosine_loss

    return total_loss


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
            distill_weight=2  # 新增参数：蒸馏权重
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
        
    def forward(self, imgs, bboxes, labels, center_points, point_labels, target_labels, original_input_size, resized_input_size, coco_image_names):
        device = imgs.device
        imgs.to("cpu")
        assert self.T_model.image_encoder.img_size == self.S_model.image_encoder.img_size
        # imgs = random_resize(imgs)

        input_images = torch.stack([self.T_model.preprocess(imgs[i,:,:,:]) for i in range(self.batch_size)], dim=0) #G padding
        input_images.to(device)
        try:
            with autocast():
                T_features, T_layers_features = self.T_model.image_encoder(input_images)
                S_features, S_layers_features = self.S_model.image_encoder(input_images)

                RATE = self.current_epoch / self.max_steps
                distill_loss = feature_distillation_loss(T_features, S_features, T_layers_features, S_layers_features, RATE)

                del T_layers_features, S_layers_features
                num_masks = sum([len(b) for b in bboxes])

                BS_LOSS = 0
                BS_loss_IoU, BS_loss_dice, BS_loss_focal = 0, 0, 0
                BS_IoU = 0
                BS_dice = 0
                if self.multimask:
                    num_masks *= 3

                for T_feature, S_feature, bbox, label, center_point, point_label, target_label, original_input_size_item, resized_input_size_item, coco_image_name, img in \
                        zip(T_features, S_features, bboxes, labels, center_points, point_labels, target_labels, original_input_size, resized_input_size, coco_image_names, imgs):
                    '''
                    original_input_size_item: ori阶段图像尺寸 => H0,W0
                    resized_input_size_item: middle阶段图像尺寸 => H1,W1(未填充成正方形)
                    img: 填充成正方形(resized_input_size_item的长边)
                    '''
                    if random.random() < 0.5:
                        self.use_bbox  = False
                    else:
                        self.use_bbox  = True
                    if self.use_bbox:
                        bbox_input=bbox
                    else:
                        bbox_input=None
                    
                    
                    sparse_embeddings, dense_embeddings = self.T_model.prompt_encoder(
                        points=(center_point, point_label),  #
                        boxes=bbox_input,
                        masks=None,
                    )
                    torch.cuda.empty_cache()

                    # Predict masks
                    low_res_masks, iou_predictions = self.T_model.mask_decoder(
                        image_embeddings=S_feature.unsqueeze(0),
                        image_pe=self.T_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=self.multimask,
                    )
                    torch.cuda.empty_cache()
                    
                    # Upscale the masks to the original image resolution
                    masks = F.interpolate(
                        low_res_masks,
                        (self.T_model.image_encoder.img_size, self.T_model.image_encoder.img_size),
                        mode="bilinear",
                        align_corners=False,
                    ) ## self.T_model.image_encoder.img_size在vit_t设置下为1024
                    torch.cuda.empty_cache()
                    del low_res_masks

                    resized_height, resized_width = img.shape[1], img.shape[2]
                    masks = masks[..., : resized_height, : resized_width]
                    # masks = F.interpolate(masks, original_input_size_item, mode="bilinear", align_corners=False)
                    torch.cuda.empty_cache()
                    # masks = masks > self.T_model.mask_threshold
                    #G 返回二值化后的数据


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
                    loss_dict = calculate_segmentation_losses(pred_masks, iou_label)
                    
                    single_img_loss_IoU = loss_dict['iou_loss']
                    single_img_loss_dice = loss_dict['dice_loss']
                    single_img_loss_focal = loss_dict['focal_loss']
                    single_img_iou = loss_dict['iou']
                    single_img_dice = loss_dict['dice']
                    
                    ext_name = coco_image_name.split('.')[-1]
                    step = self.current_epoch
                    training_visual_path = f"/data2/wuxinrui/RA-L/MobileSAM/training_visual_distill/{step}"
                    if not os.path.exists(training_visual_path):
                        os.makedirs(training_visual_path, exist_ok=True)
                    #G 1% to save images
                    # if random.random() < 0.03:
                    output_path_combined = os.path.join(training_visual_path, f"{step}_{coco_image_name.replace(ext_name, '_combined.jpg')}")


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
                    
                    # penalty_coefficient = 0.5 * (1 + single_img_iou)
                    penalty_coefficient = np.log2(1.05 + single_img_iou)
                    single_LOSS = (15 * single_img_loss_focal + 1.5 * single_img_loss_dice + 2.5 * single_img_loss_IoU) / penalty_coefficient if single_img_iou > 0.25 else 0
                    #g 排除低质样本的副作用
                    BS_LOSS += single_LOSS
                    BS_loss_IoU += single_img_loss_IoU
                    BS_loss_dice += single_img_loss_dice
                    BS_loss_focal += single_img_loss_focal
                    
                    BS_IoU += single_img_iou
                    BS_dice += single_img_dice
                    
            BS = len(imgs)
            av_BS_IoU = BS_IoU / BS
            av_BS_dice = BS_dice / BS
            av_BS_loss_IoU = BS_loss_IoU / BS
            av_BS_loss_dice = BS_loss_dice / BS
            av_BS_loss_focal = BS_loss_focal / BS
                    
                    
            return {
                'av_BS_IoU': av_BS_IoU,
                'av_BS_dice': av_BS_dice,
                'loss': (BS_LOSS + distill_loss * self.distill_weight),
                'av_BS_loss_focal': av_BS_loss_focal,
                'av_BS_loss_dice': av_BS_loss_dice,
                'av_BS_loss_IoU': av_BS_loss_IoU,
                'distill_loss': distill_loss,
            }

        except Exception as e:
            # 获取图片名称
            img_names = [coco_image_name for coco_image_name in coco_image_names]
            print(f"Error occurred while processing images: {img_names}")
            print(f"Error details: {str(e)}")
            raise e
        
    def training_step(self, batch, batch_nb):
        imgs, bboxes, labels, center_points, point_labels, img_name,category_ids ,original_input_size,resized_input_size, coco_image_names = batch


        with autocast():
            outputs = self(imgs, bboxes, labels, center_points, point_labels,category_ids,original_input_size,resized_input_size, coco_image_names)

        if not outputs.get('valid', True):
            print("Skipping invalid batch")
            return None  # 跳过当前批次
        metrics = outputs
        self.log_dict(metrics, prog_bar=True, rank_zero_only=True)
        del outputs
        torch.cuda.empty_cache()
        return metrics
