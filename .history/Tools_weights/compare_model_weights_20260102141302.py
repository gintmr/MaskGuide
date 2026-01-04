'''
比较当前模型与使用的权重之间的差异
'''

import os
import torch
from mobile_sam import sam_model_registry
os.environ['INFERENCE_MODE'] = "test"

init_model = sam_model_registry["tiny_msam"]
model = init_model(checkpoint="/data2/wuxinrui/Distill-SAM/trained_models/Distilled_encoder/msam_mix_data_1epoch.pth")
