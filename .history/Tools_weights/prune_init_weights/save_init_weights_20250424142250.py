import os
import torch
from mobile_sam import sam_model_registry
os.environ['INFERENCE_MODE'] = "test"

init_model = sam_model_registry["tiny_msam"]
model = init_model()
state_dict = model.state_dict()
torch.save(model.state_dict(), "/data2/wuxinrui/RA-L/MobileSAM/Tools_weights/prune_init_weights/init_weights_tiny_msam.pth")