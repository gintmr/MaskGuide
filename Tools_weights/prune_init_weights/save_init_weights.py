import os
import torch
import sys
module_path = "./"
if module_path not in sys.path:
    sys.path.append(module_path)

from mobile_sam import sam_model_registry
os.environ['INFERENCE_MODE'] = "test"
os.environ['MODEL_MODE'] = "test"
init_model = sam_model_registry["rep_sam"]
model = init_model()
state_dict = model.state_dict()
torch.save(model.state_dict(), "/data2/wuxinrui/RA-L/MobileSAM/Tools_weights/prune_init_weights/init_rep_sam.pth")