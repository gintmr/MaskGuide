import torch

def trans_ckpt(ckpt_path):
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

    torch.save(save_checkpoint, "/data2/wuxinrui/RA-L/MobileSAM/weights/temp_weights/temp.pth")
    return "/data2/wuxinrui/RA-L/MobileSAM/weights/temp_weights/temp.pth"

trans_ckpt()