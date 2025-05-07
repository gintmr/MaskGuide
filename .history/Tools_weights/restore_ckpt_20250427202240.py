import torch

# 加载检查点
ckpt_path = "/data2/wuxinrui/RA-L/MobileSAM/trained_models/Distilled_encoder/last-v18.ckpt"
checkpoint = torch.load(ckpt_path, map_location='cpu')

# 遍历优化器的状态并将state_steps移动到CPU
if 'optimizer_states' in checkpoint:
    for state in checkpoint['optimizer_states']:
        if 'state' in state:
            for param_state in state['state'].values():
                if 'step' in param_state:
                    param_state['step'] = param_state['step'].cpu()

# 保存修改后的检查点
torch.save(checkpoint, ckpt_path)