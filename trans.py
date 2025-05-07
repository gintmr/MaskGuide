import torch

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
        torch.save(save_checkpoint, "/data2/wuxinrui/RA-L/MobileSAM/weights/temp_weights/temp.pth")
        return "/data2/wuxinrui/RA-L/MobileSAM/weights/temp_weights/temp.pth"
    else:
        torch.save(save_checkpoint, save_path)
        return save_path

trans_ckpt("/data2/wuxinrui/RA-L/MobileSAM/trained_models/Img_Encoder_T_vit_t_S_tiny_msam/last-v5.ckpt", "/data2/wuxinrui/RA-L/MobileSAM/trained_models/Img_Encoder_T_vit_t_S_tiny_msam/only_distill40epoch_coco6epoch.pth")