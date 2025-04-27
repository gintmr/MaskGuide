import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class OceanSegmentationLoss(nn.Module):
    def __init__(self, 
                 alpha_focal=0.25, 
                 gamma_focal=2.0,
                 lambda_iou=0.4,
                 lambda_dice=0.3,
                 lambda_focal=0.2,
                 lambda_edge=0.1,
                 compute_metrics=True):
        """
        改进版海洋实例分割损失类
        新增参数:
            compute_metrics: 是否在forward中计算评估指标
        """
        super().__init__()
        self.alpha = alpha_focal
        self.gamma = gamma_focal
        self.lambda_iou = lambda_iou
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.lambda_edge = lambda_edge
        self.compute_metrics = compute_metrics
        
        # Sobel核注册为不可训练参数
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
        集成指标计算的完整前向传播
        返回:
            dict: 包含losses和metrics的字典结构
        """
        # 输入验证
        assert pred_masks.ndim in [3,4], "输入维度应为3D或4D"
        
        # 概率转换
        pred_probs = self._get_probs(pred_masks)
        ori_masks = ori_masks.float()
        
        # 结果容器
        results = {
            'losses': {},
            'metrics': {}
        }
        
        # 1. 计算所有损失项
        self._compute_losses(results, pred_masks, pred_probs, ori_masks, transition_mask)
        
        # 2. 计算评估指标（不参与梯度计算）
        if self.compute_metrics:
            with torch.no_grad():
                self._compute_metrics(results, pred_probs, ori_masks)
        
        # 3. 组合总损失
        results['losses']['total'] = self._combine_losses(results['losses'])
        
        return results
    
    def _get_probs(self, pred_masks):
        """统一处理二分类/多分类概率转换"""
        if pred_masks.ndim == 4:
            return torch.softmax(pred_masks, dim=1)[:, 1]  # 多分类取前景概率
        return torch.sigmoid(pred_masks)  # 二分类直接sigmoid
    
    def _compute_losses(self, results, pred_masks, pred_probs, ori_masks, transition_mask):
        """核心损失计算"""
        # MSIoU/IoU损失
        intersection = (pred_probs * ori_masks)
        if transition_mask is not None:
            intersection = intersection * transition_mask
            union = ((pred_probs + ori_masks) * transition_mask).sum(dim=(1,2)) - intersection.sum(dim=(1,2))
            results['losses']['msiou'] = 1 - ((intersection.sum(dim=(1,2)) + 1e-6) / (union + 1e-6)).mean()
        else:
            union = pred_probs.sum(dim=(1,2)) + ori_masks.sum(dim=(1,2)) - intersection.sum(dim=(1,2))
            results['losses']['iou'] = 1 - ((intersection.sum(dim=(1,2)) + 1e-6) / (union + 1e-6)).mean()
        
        # Dice损失
        numerator = 2 * intersection.sum(dim=(1,2)) + 1e-6
        denominator = pred_probs.sum(dim=(1,2)) + ori_masks.sum(dim=(1,2)) + 1e-6
        results['losses']['dice'] = 1 - (numerator / denominator).mean()
        
        # Focal Loss
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_masks.squeeze(1) if pred_masks.ndim==4 else pred_masks, 
            ori_masks, 
            reduction='none')
        pt = torch.exp(-bce_loss)
        results['losses']['focal'] = (self.alpha * (1-pt)**self.gamma * bce_loss).mean()
        
        # 边缘损失
        pred_edge = self._sobel_edges(pred_probs.unsqueeze(1))
        target_edge = self._sobel_edges(ori_masks.unsqueeze(1))
        results['losses']['edge'] = F.mse_loss(pred_edge, target_edge)
    
    def _compute_metrics(self, results, pred_probs, ori_masks):
        """指标计算逻辑"""
        pred_binary = (pred_probs > 0.5).long()
        ori_np = ori_masks.cpu().numpy()
        
        # 批量计算Dice/IoU
        batch_dice, batch_iou = [], []
        for p, t in zip(pred_binary.cpu().numpy(), ori_np):
            inter = np.sum((p == 1) & (t == 1))
            union = np.sum((p == 1) | (t == 1))
            
            batch_iou.append(inter / (union + 1e-6))
            batch_dice.append(2*inter / (np.sum(p) + np.sum(t) + 1e-6))
        
        results['metrics'].update({
            'dice': torch.tensor(np.mean(batch_dice)),
            'iou': torch.tensor(np.mean(batch_iou)),
            'edge_ratio': (pred_binary.sum() / (pred_binary.numel() + 1e-6)).item()  # 前景占比
        })
    
    def _combine_losses(self, losses):
        """动态权重组合"""
        iou_key = 'msiou' if 'msiou' in losses else 'iou'
        return (
            self.lambda_iou * losses[iou_key] +
            self.lambda_dice * losses['dice'] +
            self.lambda_focal * losses['focal'] +
            self.lambda_edge * losses['edge']
        )
    
    def _sobel_edges(self, x):
        """带padding的Sobel边缘检测"""
        pad_x = F.pad(x, (1,1,1,1), mode='reflect')
        grad_x = F.conv2d(pad_x, self.sobel_kernel_x)
        grad_y = F.conv2d(pad_x, self.sobel_kernel_y)
        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)  # 数值稳定