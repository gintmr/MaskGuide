import os
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import numpy as np
from PIL import Image
import cv2
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
from torch.cuda.amp import autocast, GradScaler
import logging
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import CosineAnnealingLR
import math

NUM_WORKERS=4
NUM_GPUS = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))



def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def all_gather(data, gather_batch_size=500):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    
    torch.cuda.empty_cache()
    
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # 将数据分批
    data_batches = [data[i:i + gather_batch_size] for i in range(0, len(data), gather_batch_size)]
    gathered_batches = []

    for batch in data_batches:
        # 序列化当前批次的数据
        buffer = pickle.dumps(batch)
        storage = torch.ByteStorage.from_buffer(buffer)
        tensor = torch.ByteTensor(storage).to("cuda")

        # 获取当前批次的大小
        local_size = torch.LongTensor([tensor.numel()]).to("cuda")
        size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
        dist.all_gather(size_list, local_size)
        size_list = [int(size.item()) for size in size_list]
        max_size = max(size_list)

        # 准备接收张量
        tensor_list = [torch.ByteTensor(size=(max_size,)).to("cuda") for _ in range(world_size)]

        # 填充当前批次的数据
        if local_size != max_size:
            padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
            tensor = torch.cat((tensor, padding), dim=0)

        # 执行 all_gather
        dist.all_gather(tensor_list, tensor)

        # 反序列化收集到的数据
        batch_data_list = []
        for size, t in zip(size_list, tensor_list):
            buffer = t.cpu().numpy().tobytes()[:size]
            batch_data_list.append(pickle.loads(buffer))

        # 合并批次数据
        gathered_batches.append(batch_data_list)

    # 将所有批次的数据合并为一个列表
    gathered_data = [item for batch in gathered_batches for sublist in batch for item in sublist]
    return gathered_data

class AbstractDistillFinetuner(pl.LightningModule, ABC):
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
            learning_rate=1e-4,
            weight_decay=1e-4,
            train_dataset=None,
            val_dataset=None,
            metrics_interval=10,
            multimask=False,
            use_bbox=False,
    ):
        super(AbstractDistillFinetuner, self).__init__()

        self.T_model = T_model
        self.T_model = self.load_model(T_model, T_checkpoint_path)
        self.T_model.to(device=self.device)

        self.S_model = S_model
        self.S_model = self.load_model(S_model, S_checkpoint_path)
        self.S_model.to(device=self.device)

        self.freeze_layers(freeze_image_encoder, freeze_prompt_encoder, freeze_mask_decoder)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.multimask = multimask

        self.train_metric = defaultdict(lambda: deque(maxlen=metrics_interval))
        self.use_bbox = use_bbox
        self.metrics_interval = metrics_interval

    def load_model(self, model_name, checkpoint_path):
        """
        Load a model from the model registry.
        """
        return sam_model_registry[model_name](checkpoint=checkpoint_path)

    def freeze_layers(self, freeze_image_encoder, freeze_prompt_encoder, freeze_mask_decoder):
        """
        Freeze layers based on the provided flags.
        """
        # for model in [self.T_model, self.S_model]:
        for model in [self.S_model]:
            if freeze_image_encoder:
                for param in model.image_encoder.parameters():
                    param.requires_grad = False
            if freeze_prompt_encoder:
                for param in model.prompt_encoder.parameters():
                    param.requires_grad = False
            if freeze_mask_decoder:
                for param in model.mask_decoder.parameters():
                    param.requires_grad = False
        for model in [self.T_model]:
            for param in model.image_encoder.parameters():
                param.requires_grad = False
            for param in model.prompt_encoder.parameters():
                param.requires_grad = False
            for param in model.mask_decoder.parameters():
                param.requires_grad = False


    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass of the model.
        """
        pass

    def training_step(self, batch, batch_nb):
        """
        Training step.
        """
        pass

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, capturable=True)

        # 新学习率调度函数：余弦衰减（初始→1/10）
        def lr_schedule(step):
            progress = step / self.max_steps
            #g === 先高后低
            if progress < 0.5:
                factor = 2 * progress
                decay_factor = 0.25 + 0.75 * (1 - math.cos(math.pi * factor)) / 2
            else:
                factor = 2 * (progress - 0.5)
                decay_factor = 0.25 + 0.75 * (1 + math.cos(math.pi * factor)) / 2
            
            #g ====从高到低
            # factor = progress
            # decay_factor = 0.25 + 0.75 * (1 + math.cos(math.pi * factor)) / 2

            return decay_factor
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule)
        
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': "step",
                'frequency': 1,
            }
        }

    def train_dataloader(self):
        """
        Training dataloader.
        """
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            collate_fn=self.train_dataset.collate_fn,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            shuffle=True)
        return train_loader

    def val_dataloader(self):
        """
        Validation dataloader.
        """
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            collate_fn=self.val_dataset.collate_fn,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            shuffle=False)
        return val_loader

    def feature_distillation_loss(self, T_features, S_features, RATE, reduction='batchmean', alpha=0.8, beta=0.2):
        """
        计算教师模型和学生模型特征的蒸馏损失。

        参数:
            T_features (torch.Tensor): 教师模型的特征输出。
            S_features (torch.Tensor): 学生模型的特征输出。
            T_layers_features (dict): 教师模型的各层特征输出。
            S_layers_features (dict): 学生模型的各层特征输出。
            reduction (str): 损失的归约方式，可选 'mean' 或 'sum'。
            alpha (float): KL 散度损失的权重。
            beta (float): 余弦相似度损失的权重。

        返回:
            torch.Tensor: 蒸馏损失。
        """
        epsilon = 1e-7

        if torch.isnan(S_features).any():
            S_features = torch.where(torch.isnan(S_features), torch.tensor(epsilon, device=S_features.device), S_features)
            
        temperature = 0.8 + 3.2 * (1 + math.cos(math.pi * RATE)) / 2
        temperature = max(0.8, min(2.0, temperature))  # 限制温度在 0.8 到 2.0 之间

        epsilon = 1e-7
        student_softmax = F.softmax(S_features / temperature, dim=1).clamp(min=epsilon, max=1 - epsilon)
        teacher_softmax = F.softmax(T_features / temperature, dim=1).clamp(min=epsilon, max=1 - epsilon)
        
        assert not torch.isnan(teacher_softmax).any(), "teacher_softmax contains NaN"
        assert not torch.isinf(teacher_softmax).any(), "T_features contains Inf"
        assert not torch.isnan(student_softmax).any(), "student_softmax contains NaN"
        assert not torch.isinf(student_softmax).any(), "student_softmax contains Inf"
        
        # kl_div = F.kl_div(student_softmax.log(), teacher_softmax, reduction=reduction) * (temperature ** 2)
        kl_div = torch.sum(teacher_softmax * torch.log(teacher_softmax / student_softmax), dim=1) * (temperature ** 2)
        kl_div = torch.mean(kl_div)
        assert not torch.isnan(kl_div), "KL divergence is NaN"
        assert not torch.isinf(kl_div), "KL divergence is inf"

        T_features_norm = F.normalize(T_features.detach(), p=2, dim=1)
        S_features_norm = F.normalize(S_features, p=2, dim=1)
        cosine_similarity = F.cosine_similarity(S_features_norm, T_features_norm, dim=1)
        cosine_loss = 1 - cosine_similarity.mean()

        total_loss = alpha * kl_div + beta * cosine_loss

        progress = RATE
        if progress < 0.5:  # 前期侧重特征对齐
            A, B = 0.1, 0.04
        else:  # 后期加强注意力对齐
            A, B = 0.08, 0.02

        # layers_kl_div_list = []
        # for T_layers_feature, S_layers_feature in zip(T_layers_features.values(), S_layers_features.values()):
        #     if S_layers_feature.shape[-1:] != T_layers_feature.shape[-1:]:
        #         S_layers_feature = S_layers_feature.unsqueeze(-1)
        #         T_layers_feature = T_layers_feature.unsqueeze(-1)
        #         if torch.isnan(S_layers_feature).any():
        #             S_layers_feature = torch.where(torch.isnan(S_layers_feature), torch.tensor(epsilon, device=S_layers_feature.device), S_layers_feature)
                
        #         S_layers_feature = F.interpolate(S_layers_feature, size=T_layers_feature.shape[-2:], mode='bilinear', align_corners=False)
        #         S_layers_feature = S_layers_feature.squeeze(-1)
        #         T_layers_feature = T_layers_feature.squeeze(-1)

        #     s_layer_softmax = F.softmax(S_layers_feature / temperature, dim=1).clamp(min=epsilon, max=1 - epsilon)
        #     t_layer_softmax = F.softmax(T_layers_feature / temperature, dim=1).clamp(min=epsilon, max=1 - epsilon)
        #     layers_kl_div = F.kl_div(s_layer_softmax.log(), t_layer_softmax, reduction=reduction) * (temperature ** 2)
        #     layers_kl_div = torch.sum(t_layer_softmax * torch.log(t_layer_softmax / s_layer_softmax), dim=1) * (temperature ** 2)
        #     layers_kl_div = torch.mean(layers_kl_div)
        #     assert not torch.isnan(layers_kl_div), "KL divergence is NaN"
        #     assert not torch.isinf(layers_kl_div), "KL divergence is inf"
        #     layers_kl_div_list.append(layers_kl_div)
            
        #     layers_cosine_loss = 1 - F.cosine_similarity(F.normalize(S_layers_feature, p=2, dim=1), F.normalize(T_layers_feature.detach(), p=2, dim=1), dim=1).mean()
        #     total_loss += B * layers_cosine_loss
            
        # layers_kl_div_list = torch.stack(layers_kl_div_list)
        # total_loss += A * torch.mean(layers_kl_div_list)

        return total_loss