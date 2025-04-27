import torch
import torch.nn.functional as F
import numpy as np

class OceanSegmentationLoss:
    def __init__(self, include_dice_iou=True, gamma=2.0, alpha=0.25):
        """
        初始化分割损失计算器
        
        参数:
            include_dice_iou: 是否计算Dice和IoU指标
            gamma: Focal Loss的gamma参数
            alpha: Focal Loss的alpha参数
        """
        self.include_dice_iou = include_dice_iou
        self.gamma = gamma
        self.alpha = alpha
        
    def _calculate_metrics(self, pred_binary, ori_masks):
        """计算Dice和IoU指标（不参与梯度计算）"""
        with torch.no_grad():
            dice_list, iou_list = [], []
            for p, t in zip(pred_binary.cpu().numpy(), ori_masks.cpu().numpy()):
                TP = np.sum((p == 1) & (t == 1))
                FP = np.sum((p == 1) & (t == 0))
                TN = np.sum((p == 0) & (t == 0))
                FN = np.sum((p == 0) & (t == 1))
                
                intersection = TP
                union = TP + FP + FN
                iou = intersection / (union + 1e-6)
                dice = (2*intersection) / (np.sum(p) + np.sum(t) + 1e-6)
                
                dice_list.append(dice)
                iou_list.append(iou)
            
            return torch.tensor(np.mean(dice_list)), torch.tensor(np.mean(iou_list))
    
    def _calculate_iou_loss(self, pred_probs, ori_masks):
        """计算IoU损失"""
        intersection = (pred_probs * ori_masks).sum(dim=(1,2))
        union = (pred_probs + ori_masks).sum(dim=(1,2)) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        return 1 - iou.mean()
    
    def _calculate_dice_loss(self, pred_probs, ori_masks):
        """计算Dice损失"""
        intersection = (pred_probs * ori_masks).sum(dim=(1,2))
        numerator = 2 * intersection + 1e-6
        denominator = pred_probs.sum(dim=(1,2)) + ori_masks.sum(dim=(1,2)) + 1e-6
        dice = numerator / denominator
        return 1 - dice.mean()
    
    def _calculate_focal_loss(self, pred_masks, ori_masks, chunk_size=3):
        """计算Focal Loss"""
        ce_loss = 0
        for i in range(0, len(pred_masks), chunk_size):
            actual_chunk_size = min(chunk_size, len(pred_masks) - i)
            chunk_pred_masks = pred_masks[i:i+actual_chunk_size]
            chunk_ori_masks = ori_masks[i:i+actual_chunk_size].float()

            assert chunk_pred_masks.shape == chunk_ori_masks.shape, \
                f"Shapes do not match: {chunk_pred_masks.shape} vs {chunk_ori_masks.shape}"

            # 下采样以减少计算量
            chunk_pred_masks = F.interpolate(chunk_pred_masks.unsqueeze(0), 
                                           scale_factor=0.5, 
                                           mode='bilinear').squeeze(0)
            chunk_ori_masks = F.interpolate(chunk_ori_masks.unsqueeze(0), 
                                          scale_factor=0.5, 
                                          mode='bilinear').squeeze(0)
            
            ce_loss += F.binary_cross_entropy_with_logits(
                chunk_pred_masks, chunk_ori_masks, reduction='mean')
            
            del chunk_pred_masks, chunk_ori_masks
        
        # 转换为Focal Loss
        pt = torch.exp(-ce_loss)
        return self.alpha * (1 - pt) ** self.gamma * ce_loss
    
    def __call__(self, pred_masks, ori_masks):
        """
        计算分割任务的多种损失指标，保持梯度反向传播
        
        参数:
            pred_masks: 预测mask (torch.Tensor [N,C,H,W] 或 [N,H,W])
            ori_masks: 真实mask (torch.Tensor [N,H,W])
            
        返回:
            dict: 包含各项损失和指标的字典
        """
        assert isinstance(pred_masks, torch.Tensor) and isinstance(ori_masks, torch.Tensor)
        
        # 处理二分类和多分类情况
        if pred_masks.ndim == 4:  # [N,C,H,W]
            pred_probs = torch.softmax(pred_masks, dim=1)
            pred_binary = pred_probs.argmax(dim=1)
        else:  # [N,H,W]
            pred_probs = torch.sigmoid(pred_masks)
            pred_binary = (pred_probs > 0.5).long()
        
        # 初始化结果字典
        results = {}
        
        # 计算PA和IoU指标（不参与梯度计算）
        if self.include_dice_iou:
            dice, iou = self._calculate_metrics(pred_binary, ori_masks)
            results['dice'] = dice
            results['iou'] = iou
        
        # 计算各种损失
        results['iou_loss'] = self._calculate_iou_loss(pred_probs, ori_masks)
        results['dice_loss'] = self._calculate_dice_loss(pred_probs, ori_masks)
        results['focal_loss'] = self._calculate_focal_loss(pred_masks, ori_masks)
        
        # 组合损失（可根据需要调整权重）
        results['total_loss'] = (
            0.5 * results['iou_loss'] + 
            0.5 * results['dice_loss'] + 
            1.0 * results['focal_loss']
        )
        
        return results