import torch
import torch.nn as nn
import time
import torch.cuda.amp as amp
import torch.cuda.profiler as profiler
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pynvml
import sys
module_path = "./"
if module_path not in sys.path:
    sys.path.append(module_path)
from mobile_sam.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer, TinyViT, tiny_TinyViT
from mobile_sam.repvit import repvit_m0_5

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

def test_image_encoder(image_encoder, dataset, batch_size=32, num_batches=320):
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

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
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

# # 定义 TinyViT 的参数
# image_encoder = tiny_TinyViT(
#     img_size=1024,
#     in_chans=3,
#     num_classes=1000,
#     embed_dims=[64, 96, 128, 320],
#     depths=[1, 2, 4, 1],
#     num_heads=[2, 3, 4, 8],
#     window_sizes=[7, 7, 14, 7],
#     mlp_ratio=4.,
#     drop_rate=0.,
#     drop_path_rate=0.0,
#     use_checkpoint=False,
#     mbconv_expand_ratio=4.0,
#     local_conv_size=3,
#     layer_lr_decay=0.8
# )

image_encoder = tiny_TinyViT(
    img_size=1024,
    in_chans=3,
    num_classes=1000,
    # embed_dims=[64, 128, 160, 320],
    # depths=[2, 2, 6, 2],
    # num_heads=[2, 4, 5, 10],
    # window_sizes=[7, 7, 14, 7],
    # embed_dims=[64, 96, 128, 320],
    # depths=[1, 2, 4, 1],
    # num_heads=[2, 3, 4, 8],
    # window_sizes=[7, 7, 14, 7],
    embed_dims=[64, 80, 160, 320],
    depths=[1, 1, 1, 1],
    num_heads=[2, 2, 4, 8],
    window_sizes=[7, 7, 14, 7],
    mlp_ratio=4.,
    drop_rate=0.,
    drop_path_rate=0.0,
    use_checkpoint=False,
    mbconv_expand_ratio=4.0,
    local_conv_size=3,
    layer_lr_decay=0.8
)
# image_encoder = repvit_m0_5()
# 创建随机图像数据集
dataset = RandomImageDataset(size=10000, image_size=1024)

# 运行测试
test_image_encoder(image_encoder, dataset, batch_size=1, num_batches=320)