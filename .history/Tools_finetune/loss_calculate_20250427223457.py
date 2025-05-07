import torch
import torch.nn.functional as F
import numpy as np
from torchvision.ops import sigmoid_focal_loss

class OceanSegmentationLoss:
    def __init__(self, include_dice_iou=True, alpha=0.7, beta=0.3):
        """
        初始化分割损失计算器
        
        参数:
            include_dice_iou: 是否计算Dice和IoU指标
            alpha: Tversky Loss的alpha参数（惩罚FP）
            beta: Tversky Loss的beta参数（惩罚FN）
        """
        self.include_dice_iou = include_dice_iou
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
                dice = (2*intersection) / (np.sum(p) + np.sum(t) + 1e-6)
                
                dice_list.append(dice)
                iou_list.append(iou)
            
            return torch.tensor(np.mean(dice_list)), torch.tensor(np.mean(iou_list))


    def _calculate_iou_loss(self, pred_probs, ori_masks):
        torch.cuda.empty_cache() 
        """计算IoU损失"""
        intersection = (pred_probs * ori_masks).sum(dim=(1,2))
        union = (pred_probs + ori_masks).sum(dim=(1,2)) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        return 1 - iou.mean()


    def _calculate_dice_loss(self, pred_probs, ori_masks):
        torch.cuda.empty_cache() 
        """计算Dice损失"""
        intersection = (pred_probs * ori_masks).sum(dim=(1,2))
        numerator = 2 * intersection + 1e-6
        denominator = pred_probs.sum(dim=(1,2)) + ori_masks.sum(dim=(1,2)) + 1e-6
        dice = numerator / denominator
        return 1 - dice.mean()


    def _calculate_tversky_loss(self, pred_probs, ori_masks):
        torch.cuda.empty_cache() 
        """计算Tversky Loss"""
        TP = (pred_probs * ori_masks).sum(dim=(1, 2))
        FP = (pred_probs * (1 - ori_masks)).sum(dim=(1, 2))
        FN = ((1 - pred_probs) * ori_masks).sum(dim=(1, 2))
        tversky = (TP + 1e-6) / (TP + self.alpha * FP + self.beta * FN + 1e-6)
        return 1 - tversky.mean()


    def _calculate_mse_loss(self, pred_probs, ori_masks):
        torch.cuda.empty_cache() 
        """计算MSE损失"""
        mse_loss = F.mse_loss(pred_probs, ori_masks, reduction='mean')
        return mse_loss
    
    def _calculate_focal_loss(self,
        pred_logits: torch.Tensor,
        ori_masks: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "mean",
        chunk_size: int = None,
        inplace: bool = False  # 是否启用原地操作
    ) -> torch.Tensor:
        # 自动混合精度兼容
        inputs = pred_logits
        targets = ori_masks

        if torch.is_autocast_enabled():
            inputs = inputs.float()
            targets = targets.float()
        
        if chunk_size is None or inputs.numel() <= chunk_size:
            # ---- 核心计算部分 ----
            if inplace:
                p = torch.sigmoid_(inputs)  # 原地操作
            else:
                p = torch.sigmoid(inputs)
            
            ce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduction="none"
            )
            p_t = p * targets + (1 - p) * (1 - targets)
            loss = ce_loss * ((1 - p_t) ** gamma)
            
            if alpha >= 0:
                alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
                loss = alpha_t * loss
            
            # 归约操作
            if reduction == "mean":
                return loss.mean()
            elif reduction == "sum":
                return loss.sum()
            return loss
        
        else:
            # ---- 分块计算部分 ----
            loss_sum = 0
            num_elements = inputs.numel()
            num_chunks = (num_elements + chunk_size - 1) // chunk_size
            
            for i in range(num_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, num_elements)
                
                # 分块索引（保持原始形状）
                indices = torch.arange(start, end, device=inputs.device)
                chunk_inputs = inputs.view(-1)[indices].view_as(inputs)
                chunk_targets = targets.view(-1)[indices].view_as(targets)
                
                # 递归计算分块
                chunk_loss = sigmoid_focal_loss(
                    chunk_inputs, chunk_targets, 
                    alpha, gamma, reduction, None, inplace
                )
                
                # 累加结果
                if reduction == "sum":
                    loss_sum += chunk_loss
                else:
                    loss_sum += chunk_loss * (end - start) / num_elements
            
            return loss_sum

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
        pred_masks = pred_masks.float()
        ori_masks = ori_masks.float()
        
        # 处理多分类 vs 二分类
        if pred_masks.ndim == 4:  # [N,C,H,W] 多分类
            pred_logits = pred_masks
            pred_probs = torch.softmax(pred_masks, dim=1)  # 多分类归一化
            pred_binary = pred_probs.argmax(dim=1)          # 二值化
        else:  # [N,H,W] 二分类
            pred_logits = pred_masks
            pred_probs = torch.sigmoid(pred_masks)          # 二分类归一化
            pred_binary = (pred_probs > 0.5).long()         # 二值化
        
        # 初始化结果字典
        results = {}
        
        # 计算PA和IoU指标（不参与梯度计算）
        if self.include_dice_iou:
            dice, iou = self._calculate_metrics(pred_binary, ori_masks)
            results['dice'] = dice
            results['iou'] = iou

        # 损失计算（明确输入类型）
        results.update({
            'iou_loss': self._calculate_iou_loss(pred_probs, ori_masks),         # 需归一化概率
            'dice_loss': self._calculate_dice_loss(pred_probs, ori_masks),       # 需归一化概率
            'tversky_loss': self._calculate_tversky_loss(pred_probs, ori_masks), # 需归一化概率
            'focal_loss': self._calculate_focal_loss(pred_logits, ori_masks, chunk_size=128, reduction="mean"),            # 需原始logits
            'mse_loss': F.mse_loss(pred_probs, ori_masks)                         # 需归一化概率
        })
        # # 计算各种损失
        # results['iou_loss'] = self._calculate_iou_loss(pred_probs, ori_masks)
        # results['dice_loss'] = self._calculate_dice_loss(pred_probs, ori_masks)
        # results['tversky_loss'] = self._calculate_tversky_loss(pred_probs, ori_masks)
        # results['focal_loss'] = self._calculate_focal_loss(pred_probs, ori_masks)
        # results['mse_loss'] = self._calculate_mse_loss(pred_probs, ori_masks)

        # 组合损失（可根据需要调整权重）
        total_loss = (
            0.3 * results['iou_loss'] + 
            0.5 * results['dice_loss'] + 
            0.3 * results['tversky_loss'] +
            20.0 * results['focal_loss'] + 
            1.0 * results['mse_loss']
        )

        penalty_coefficient = np.log2(1.05 + iou)
        total_loss = total_loss / penalty_coefficient if iou > 0.25 else total_loss / 0.2
        results['total_loss'] = total_loss

        return results