'''
用于比较两个权重文件的差异
'''
import torch

def format_number(num):
    """格式化数字，以千分位分隔"""
    return f"{num:,}"

def compare_model_weights(state_dict1, state_dict2):
    """
    比较两个模型的权重参数文件，输出模型大小比较和权重差异的层。
    :param state_dict1: 第一个模型的 state_dict
    :param state_dict2: 第二个模型的 state_dict
    """
    # 初始化字典用于存储每个模型的参数量和内存占用
    module_params1 = {}
    module_memory1 = {}
    module_params2 = {}
    module_memory2 = {}

    # 遍历第一个模型的 state_dict
    for key, param in state_dict1.items():
        module_name = key.split('.')[0]
        if module_name not in module_params1:
            module_params1[module_name] = 0
            module_memory1[module_name] = 0
        num_elements = param.numel()
        module_params1[module_name] += num_elements
        bytes_per_element = torch.finfo(param.dtype).bits // 8 if param.dtype.is_floating_point else torch.iinfo(param.dtype).bits // 8
        memory_size_bytes = num_elements * bytes_per_element
        module_memory1[module_name] += memory_size_bytes

    # 遍历第二个模型的 state_dict
    for key, param in state_dict2.items():
        module_name = key.split('.')[0]
        if module_name not in module_params2:
            module_params2[module_name] = 0
            module_memory2[module_name] = 0
        num_elements = param.numel()
        module_params2[module_name] += num_elements
        bytes_per_element = torch.finfo(param.dtype).bits // 8 if param.dtype.is_floating_point else torch.iinfo(param.dtype).bits // 8
        memory_size_bytes = num_elements * bytes_per_element
        module_memory2[module_name] += memory_size_bytes

    # 比较两个模型的总参数量和内存占用
    total_params1 = sum(module_params1.values())
    total_memory1 = sum(module_memory1.values())
    total_params2 = sum(module_params2.values())
    total_memory2 = sum(module_memory2.values())

    print("模型大小比较：")
    print(f"模型1总参数量: {format_number(total_params1)}")
    print(f"模型1总内存占用: {total_memory1 / (1024 * 1024):.2f} MB")
    print(f"模型2总参数量: {format_number(total_params2)}")
    print(f"模型2总内存占用: {total_memory2 / (1024 * 1024):.2f} MB")

    if total_params1 > total_params2:
        print("模型1更大")
    elif total_params1 < total_params2:
        print("模型2更大")
    else:
        print("两个模型大小相同")

    # 找出权重差异的层
    print("\n权重差异的层：")
    diff_layers = []
    for key in state_dict1.keys():
        if key not in state_dict2:
            print(f"层 {key} 仅存在于模型1")
            diff_layers.append(key)
        elif not torch.equal(state_dict1[key], state_dict2[key]):
            print(f"层 {key} 权重不同")
            diff_layers.append(key)

    for key in state_dict2.keys():
        if key not in state_dict1:
            print(f"层 {key} 仅存在于模型2")
            diff_layers.append(key)

    if not diff_layers:
        print("两个模型权重完全相同")

# 加载两个模型的 state_dict
state_dict_path1 = '/data2/wuxinrui/RA-L/MobileSAM/weights/weights_prune_init/init_weights_tiny_msam.pth'  # 替换为第一个模型的权重文件路径
state_dict_path2 = '/data2/wuxinrui/RA-L/MobileSAM/trained_models/Distilled_encoder/COCO_train_1epoch.pth'  # 替换为第二个模型的权重文件路径
state_dict1 = torch.load(state_dict_path1, map_location=torch.device('cpu'))
state_dict2 = torch.load(state_dict_path2, map_location=torch.device('cpu'))

# 比较模型权重
compare_model_weights(state_dict1, state_dict2)