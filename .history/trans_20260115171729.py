import torch
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def trans_ckpt(ckpt_path, save_path =None):
    checkpoint = torch.load(ckpt_path, map_location="cuda")
    # print(checkpoint.keys())
    state_dict = checkpoint['state_dict']
    save_checkpoint = {}
    for k, v in state_dict.items():
        if "S_model.image_encoder" in k:
            save_checkpoint[k.replace("S_model.image_encoder", "image_encoder")] = v
        elif "T_model.prompt_encoder" in k:
            save_checkpoint[k.replace("T_model.prompt_encoder", "prompt_encoder")] = v
        elif "T_model.mask_decoder" in k:
            save_checkpoint[k.replace("T_model.mask_decoder", "mask_decoder")] = v

    if not save_path:
        torch.save(save_checkpoint, "/data2/wuxinrui/Distill-SAM/weights/temp_weights/temp.pth")
        return "/data2/wuxinrui/Distill-SAM/weights/temp_weights/temp.pth"
    else:
        torch.save(save_checkpoint, save_path)
        return save_path
import torch

def compare_mask_decoders(ckpt_path, reference_path):
    # 加载当前 checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cuda")
    state_dict = checkpoint['state_dict']
    
    # 加载参考模型
    reference_state_dict = torch.load(reference_path, map_location="cuda")
    # reference_state_dict = reference_checkpoint['state_dict']
    
    # 提取 T_model.mask_decoder 和 S_model.mask_decoder 的权重
    t_mask_decoder_weights = {k: v for k, v in state_dict.items() if "T_model.mask_decoder" in k}
    s_mask_decoder_weights = {k: v for k, v in state_dict.items() if "S_model.mask_decoder" in k}
    
    # 提取 reference 中的 mask_decoder 权重
    reference_mask_decoder_weights = {k: v for k, v in reference_state_dict.items() if "mask_decoder" in k}
    
    # 比较 T_model.mask_decoder 和 reference 中的 mask_decoder 权重
    t_match = all(torch.equal(t_mask_decoder_weights[k.replace("mask_decoder", "T_model.mask_decoder")], v) for k, v in reference_mask_decoder_weights.items())
    
    # 比较 S_model.mask_decoder 和 reference 中的 mask_decoder 权重
    s_match = all(torch.equal(s_mask_decoder_weights[k.replace("mask_decoder", "S_model.mask_decoder")], v) for k, v in reference_state_dict.items() if "mask_decoder" in k)
    
    # 输出比较结果
    if t_match:
        print("T_model.mask_decoder 的权重与 reference 中的 mask_decoder 权重相同。")
    else:
        print("T_model.mask_decoder 的权重与 reference 中的 mask_decoder 权重不同。")
    
    if s_match:
        print("S_model.mask_decoder 的权重与 reference 中的 mask_decoder 权重相同。")
    else:
        print("S_model.mask_decoder 的权重与 reference 中的 mask_decoder 权重不同。")

# # 调用函数
# compare_mask_decoders(
#     "/data2/wuxinrui/Distill-SAM/trained_models/Img_Encoder_T_vit_t_S_tiny_msam/last-v13.ckpt",
#     "/data2/wuxinrui/Distill-SAM/weights/mobile_sam.pt"
# )



def trans_pth(pth_path, reference_path=None):
    T_model_path = "/data2/wuxinrui/Distill-SAM/weights/mobile_sam.pt"

    pth_weights = torch.load(pth_path, map_location="cuda")
    T_weights = torch.load(T_model_path, map_location="cuda")

    for k, v in pth_weights.items():
        if "mask_decoder" in k:
            pth_weights[k] = T_weights[k]
    if not reference_path:
        reference_path = pth_path.replace(".pth", "_mobilesamMaskdecoder.pth")
    torch.save(pth_weights, reference_path)



# compare_mask_decoders_and_image_encoders("path_to_ckpt.pth", "path_to_reference.pth")
# trans_ckpt("ori.ckpt", "target.pth")
# trans_pth("target.pth")
