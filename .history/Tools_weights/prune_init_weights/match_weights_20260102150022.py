import torch
import torch.nn.init as init

pruned_state = torch.load("/data2/wuxinrui/Distill-SAM/weights/init_tiny_msam.pth")
complex_state_dict = torch.load("/data2/wuxinrui/Distill-SAM/weights/mobile_sam.pt")

def load_matched_weights(pruned_state, complex_state_dict, init_mode='current'):
    """
    智能权重加载函数
    :param pruned_state: 精简模型的权重
    :param complex_state_dict: 复杂模型的权重
    :param init_mode: 初始化模式，'current' 保持当前初始化方式，'custom' 使用自定义初始化方式
    """
    matched_keys = []
    
    if init_mode == 'current':
        # 当前初始化方式
        # 第一轮：精确匹配
        for key in pruned_state:
            if key in complex_state_dict: 
                if pruned_state[key].shape == complex_state_dict[key].shape:  # 形状匹配
                    pruned_state[key] = complex_state_dict[key]  # 直接赋值
                    matched_keys.append(key)
        
        # 第二轮：维度兼容匹配（处理embed_dim变化）
        for key in pruned_state:
            if key not in matched_keys and "weight" in key:
                if key in complex_state_dict:
                    # 检查维度是否匹配
                    if len(pruned_state[key].shape) == len(complex_state_dict[key].shape):
                        # 卷积层权重适配
                        if len(pruned_state[key].shape) == 4:  # Conv2d权重
                            min_cin = min(pruned_state[key].shape[1], complex_state_dict[key].shape[1])
                            min_cout = min(pruned_state[key].shape[0], complex_state_dict[key].shape[0])
                            pruned_state[key][:min_cout, :min_cin] = complex_state_dict[key][:min_cout, :min_cin]
                            matched_keys.append(key)
                        # 线性层权重适配        
                        elif len(pruned_state[key].shape) == 2:  # Linear层
                            min_dim2 = min(pruned_state[key].shape[1], complex_state_dict[key].shape[1])
                            min_dim1 = min(pruned_state[key].shape[0], complex_state_dict[key].shape[0])
                            pruned_state[key][:min_dim1, :min_dim2] = complex_state_dict[key][:min_dim1, :min_dim2]
                            matched_keys.append(key)
                        # 处理一维权重（例如偏置项）
                        elif len(pruned_state[key].shape) == 1:  # 一维权重
                            min_dim = min(pruned_state[key].shape[0], complex_state_dict[key].shape[0])
                            pruned_state[key][:min_dim] = complex_state_dict[key][:min_dim]
                            matched_keys.append(key)
                    else:
                        # 如果维度数目不匹配，则根据张量维度选择初始化方法
                        print(f"Initializing key {key} due to dimension mismatch.")
                        if len(pruned_state[key].shape) >= 2:
                            init.kaiming_uniform_(pruned_state[key])  # 使用 Kaiming 初始化
                        else:
                            init.zeros_(pruned_state[key])  # 对一维张量使用零初始化
                        matched_keys.append(key)
        
        # 第三轮：处理偏置项（bias）
        for key in pruned_state:
            if key not in matched_keys and "bias" in key:
                if key in complex_state_dict:
                    # 如果维度匹配，则直接赋值
                    if len(pruned_state[key].shape) == len(complex_state_dict[key].shape):
                        min_dim = min(pruned_state[key].shape[0], complex_state_dict[key].shape[0])
                        pruned_state[key][:min_dim] = complex_state_dict[key][:min_dim]
                        matched_keys.append(key)
                    else:
                        # 如果维度数目不匹配，则初始化偏置项
                        print(f"Initializing key {key} due to dimension mismatch.")
                        init.zeros_(pruned_state[key])  # 使用零初始化
                        matched_keys.append(key)
    
    elif init_mode == 'custom':
        # 自定义初始化方式
        for key in pruned_state:
            if key in complex_state_dict:
                # 对所有权重和偏置项进行自定义初始化
                if len(pruned_state[key].shape) >= 2:
                    init.xavier_uniform_(pruned_state[key])  # 使用 Xavier 初始化
                else:
                    init.constant_(pruned_state[key], 0.01)  # 对一维张量使用常量初始化
                matched_keys.append(key)
            else:
                print(f"Key {key} not found in complex_state_dict. Initializing with custom method.")
                if len(pruned_state[key].shape) >= 2:
                    init.xavier_uniform_(pruned_state[key])  # 使用 Xavier 初始化
                else:
                    init.constant_(pruned_state[key], 0.01)  # 对一维张量使用常量初始化
                matched_keys.append(key)
    
    else:
        raise ValueError("Invalid init_mode. Choose 'current' or 'custom'.")
    
    print("Matched keys:", matched_keys)
    torch.save(pruned_state, "/data2/wuxinrui/Distill-SAM/Tools_weights/prune_init_weights/init_weights_tiny_msam.pth")

# 调用函数，选择初始化模式
# load_matched_weights(pruned_state, complex_state_dict, init_mode='current')  # 使用当前初始化方式
load_matched_weights(pruned_state, complex_state_dict, init_mode='custom')  # 使用自定义初始化方式