
from .repvit import repvit_m0_5

repvit_model = repvit_m0_5(out_indices=[10])

# Test the model
import torch
import torch.nn as nn
import time
import torch.cuda.amp as amp
import torch.cuda.profiler as profiler
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pynvml

from mobile_sam.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer, TinyViT, tiny_TinyViT

# 假设你的 TinyViT 和其他组件已经定义好了
# from your_model_file import TinyViT, PromptEncoder, MaskDecoder, TwoWayTransformer

class RandomImageDataset(Dataset):
    """生成随机图像数据集"""
    def __init__(self, size, image_size):
        self.size = size
        self.image_size = image_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 生成随机图像张量
        image = torch.randn(3, self.image_size, self.image_size)
        return image

def test_image_encoder(image_encoder, dataset, batch_size=32, num_batches=32):
    # 将模型和数据移动到 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_encoder.to(device)
    image_encoder.eval()

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 初始化计时器和资源监控
    total_time = 0
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 假设使用第一块 GPU
    import itertools
    with torch.no_grad():
        for batch_idx, batch in itertools.islice(enumerate(dataloader), 5):
            if batch_idx >= num_batches:
                break

            # 将数据移动到 GPU
            batch = batch.to(device)

            # 开始计时
            start_time = time.time()

            # 启用 CUDA Profiler
            profiler.start()
            with amp.autocast(enabled=True):  # 使用混合精度加速
                output = image_encoder(batch)
            profiler.stop()

            # 结束计时
            end_time = time.time()
            total_time += (end_time - start_time)

            # 获取 GPU 使用情况
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_used = mem_info.used / (1024 ** 2)  # 转换为 MB
            print(f"Batch {batch_idx + 1}/{num_batches}, Time: {end_time - start_time:.4f}s, GPU Memory Used: {gpu_memory_used:.2f} MB")

    # 计算平均耗时
    avg_time = total_time / num_batches
    print(f"Average Time per Batch: {avg_time:.4f}s")

    # 清理
    pynvml.nvmlShutdown()


image_encoder = repvit_model
# 创建随机图像数据集
dataset = RandomImageDataset(size=1000, image_size=1024)

# 运行测试
test_image_encoder(image_encoder, dataset, batch_size=1, num_batches=10)