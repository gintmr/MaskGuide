import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class OceanSegmentationLoss(nn.Module):
    def __init__(self, 
                 alpha_focal=0.25, 
                 gamma_focal=2.0,
                 lambda_iou=0.5,
                 lambda_dice=0.5,
                 lambda_focal=0.8,
                 lambda_edge=1.2):
        """
        海洋实例分割损失计算类
        参数:
            alpha_focal: Focal Loss的类别平衡参数
            gamma_focal: Focal Loss的困难样本调节参数
            lambda_*: 各损失项的权重系数
        """
        super().__init__()
        self.alpha = alpha_focal
        self.gamma = gamma_focal
        self.lambda_iou = lambda_iou
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.lambda_edge = lambda_edge
        
        # 初始化Sobel边缘检测核
        self.register_buffer('sobel_kernel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]], dtype=torch.float32).view(1,1,3,3))
        self.register_buffer('sobel_kernel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]], dtype=torch.float32).view(1,1,3,3))

    def forward(self, pred_masks, ori_masks, transition_mask=None):
        """
        前向计算
        参数:
            pred_masks: 预测mask (N,C,H,W)或(N,H,W)
            ori_masks: 真实mask (N,H,W)
            transition_mask: 过渡带权重掩膜 (N,H,W)
        返回:
            dict: 包含各项损失和总损失的字典
        """
        # 输入验证
        assert pred_masks.ndim in [3,4], "输入维度应为3D或4D"
        assert ori_masks.shape == pred_masks.shape[-2:], "预测与真实mask尺寸不匹配"
        
        # 概率转换
        if pred_masks.ndim == 4:  # 多分类转二分类
            pred_probs = torch.softmax(pred_masks, dim=1)[:, 1]
        else:
            pred_probs = torch.sigmoid(pred_masks)
        
        ori_masks = ori_masks.float()
        results = {}
        
        # 1. MSIoU/IoU损失
        if transition_mask is not None:
            intersection = (pred_probs * ori_masks * transition_mask).sum(dim=(1,2))
            union = ((pred_probs + ori_masks) * transition_mask).sum(dim=(1,2)) - intersection
            results['msiou_loss'] = 1 - ((intersection + 1e-6) / (union + 1e-6)).mean()
        else:
            intersection = (pred_probs * ori_masks).sum(dim=(1,2))
            union = pred_probs.sum(dim=(1,2)) + ori_masks.sum(dim=(1,2)) - intersection
            results['iou_loss'] = 1 - ((intersection + 1e-6) / (union + 1e-6)).mean()
        
        # 2. Dice损失
        numerator = 2 * intersection + 1e-6
        denominator = pred_probs.sum(dim=(1,2)) + ori_masks.sum(dim=(1,2)) + 1e-6
        results['dice_loss'] = 1 - (numerator / denominator).mean()
        
        # 3. Focal Loss
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_masks.squeeze(1) if pred_masks.ndim==4 else pred_masks, 
            ori_masks, 
            reduction='none')
        pt = torch.exp(-bce_loss)
        results['focal_loss'] = (self.alpha * (1-pt)**self.gamma * bce_loss).mean()
        
        # 4. 边缘损失
        pred_edge = self._sobel_edges(pred_probs.unsqueeze(1))
        target_edge = self._sobel_edges(ori_masks.unsqueeze(1))
        results['edge_loss'] = F.mse_loss(pred_edge, target_edge)
        
        # 动态权重组合
        iou_key = 'msiou_loss' if transition_mask is not None else 'iou_loss'
        results['total_loss'] = (
            self.lambda_iou * results[iou_key] +
            self.lambda_dice * results['dice_loss'] +
            self.lambda_focal * results['focal_loss'] +
            self.lambda_edge * results['edge_loss']
        )
        
        return results
    
    def _sobel_edges(self, x):
        """Sobel边缘检测"""
        pad_x = F.pad(x, (1,1,1,1), mode='reflect')
        grad_x = F.conv2d(pad_x, self.sobel_kernel_x)
        grad_y = F.conv2d(pad_x, self.sobel_kernel_y)
        return torch.sqrt(grad_x**2 + grad_y**2)
    
    @torch.no_grad()
    def compute_metrics(self, pred, target):
        """计算评估指标(Dice/IoU)"""
        pred_binary = (pred > 0.5).long()
        target = target.cpu().numpy()
        
        dice, iou = [], []
        for p, t in zip(pred_binary.cpu().numpy(), target):
            intersection = np.sum((p == 1) & (t == 1))
            union = np.sum((p == 1) | (t == 1))
            
            iou.append(intersection / (union + 1e-6))
            dice.append(2*intersection / (np.sum(p) + np.sum(t) + 1e-6))
            
        return {
            'dice': torch.tensor(np.mean(dice)),
            'iou': torch.tensor(np.mean(iou))
        }