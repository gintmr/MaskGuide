import torch
import torch.nn.functional as F
import numpy as np

class OceanSegmentationLoss:
    def __init__(self, include_dice_iou=True, gamma=2.0, alpha=0.25, beta=0.5):
        """
        初始化分割损失计算器
        
        参数:
            include_dice_iou: 是否计算Dice和IoU指标
            gamma: Focal Loss的gamma参数
            alpha: Focal Loss的alpha参数
            beta: Tversky Loss的beta参数
        """
        self.include_dice_iou = include_dice_iou
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        
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
                dice = (2 * intersection) / (np.sum(p) + np.sum(t) + 1e-6)
                
                dice_list.append(dice)
                iou_list.append(iou)
            
            return torch.tensor(np.mean(dice_list)), torch.tensor(np.mean(iou_list))
    
    def _calculate_dice_loss(self, pred_probs, ori_masks):
        """计算Dice损失"""
        intersection = (pred_probs * ori_masks).sum(dim=(1, 2))
        numerator = 2 * intersection + 1e-6
        denominator = pred_probs.sum(dim=(1, 2)) + ori_masks.sum(dim=(1, 2)) + 1e-6
        dice = numerator / denominator
        return 1 - dice.mean()
    
    def _calculate_focal_loss(self, pred_masks, ori_masks):
        """计算Focal Loss"""
        pred_probs = torch.sigmoid(pred_masks)
        bce_loss = F.binary_cross_entropy_with_logits(pred_masks, ori_masks, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
    
    def _calculate_tversky_loss(self, pred_probs, ori_masks):
        """计算Tversky Loss"""
        TP = (pred_probs * ori_masks).sum(dim=(1, 2))
        FP = (pred_probs * (1 - ori_masks)).sum(dim=(1, 2))
        FN = ((1 - pred_probs) * ori_masks).sum(dim=(1, 2))
        tversky = TP / (TP + self.beta * FP + (1 - self.beta) * FN + 1e-6)
        return 1 - tversky.mean()
    
    def __call__(self, pred_masks, ori_masks):
        """
        计算分割任务的多种损失指标，保持梯度反向传播
        
        参数:
            pred_masks: 预测mask (torch.Tensor [N,H,W])
            ori_masks: 真实mask (torch.Tensor [N,H,W])
            
        返回:
            dict: 包含各项损失和指标的字典
        """
        assert isinstance(pred_masks, torch.Tensor) and isinstance(ori_masks, torch.Tensor)
        assert pred_masks.ndim == 3 and ori_masks.ndim == 3, "Input masks should be [N, H, W]"
        
        # 二分类情况
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
        results['dice_loss'] = self._calculate_dice_loss(pred_probs, ori_masks)
        results['focal_loss'] = self._calculate_focal_loss(pred_masks, ori_masks)
        results['tversky_loss'] = self._calculate_tversky_loss(pred_probs, ori_masks)
        
        # 组合损失（可根据需要调整权重）
        results['total_loss'] = (
            0.4 * results['dice_loss'] + 
            0.3 * results['focal_loss'] + 
            0.3 * results['tversky_loss']
        )
        
        return results