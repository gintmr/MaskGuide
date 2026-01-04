
import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1" 
import argparse
import sys
module_path = "./"
if module_path not in sys.path:
    sys.path.append(module_path)
from tqdm import tqdm
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
# module.py
from Datasets.evaluate_data import EvalDataset
from Tools_metrics.evaluate import Evaluate

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
NUM_GPUS = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
DEVICE = 'cuda'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ['INFERENCE_MODE'] = "test"# 是否下采样1/2
# os.environ['INFERENCE_MODE'] = "train"# 是否下采样1/2
os.environ['MODEL_MODE'] = "test" # 是否构造完整模型
os.environ['DISTILL'] = "F"


def main():
    parser = argparse.ArgumentParser()
    

    parser.add_argument("--test_img", default="/data2/wuxinrui/Datasets/IMC_1000/MIMC_FINAL/seen/test_list", type=str, required=False, help="path to the data root")
    parser.add_argument("--test_anno", default="/data2/wuxinrui/Datasets/IMC_1000/jsons_for_salient_instance_segmentation/test_1_prompts.json", type=str, required=False, help="path to the annotation file")
    parser.add_argument("--label_num", type=int, default=5, help="the num of points // more prior than label_path")
    
    
    parser.add_argument("--model", default='vit_t', type=str, required=False, help="model type")
    parser.add_argument("--checkpoint_path", default="/data2/wuxinrui/Distill-SAM/weights/mobile_sam.pt", type=str, required=False, help="path to the checkpoint")


    # 添加一个名为multimask的参数，类型为布尔型，默认值为False，当该参数被指定时，其值为True，用于生成多掩码
    parser.add_argument("--multimask", action="store_true", help="generate multi masks")
    # 添加一个名为use_bbox的参数，类型为布尔型，默认值为False，当该参数被指定时，其值为True，用于生成多掩码
    parser.add_argument("--use_centerpoint", action="store_true", help="use only one center point", default=False)
    parser.add_argument('--every_n_train_steps', default=50)
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--save_topk", type=int, default=3, help="save top K models")
    parser.add_argument("--image_size", type=int, default=1024, help="image size")
    parser.add_argument("--epochs", type=int, default=5, help="number of steps")
    parser.add_argument("--num_points", type=int, default=6, help="number of random points")
    parser.add_argument("--length", type=int, default=200, help="the length `of the chosen masks")
 
    args = parser.parse_args()
    
    
    
    test_dataset = EvalDataset(data_root=args.test_img, annotation_path=args.test_anno, image_size=args.image_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)
    
    Evaluate_model = Evaluate(args.model,
        args.checkpoint_path,
        batch_size=args.batch_size,
        logger=None
    )
    
    for data in tqdm(test_dataloader):
        imgs, bboxes, labels, center_points, point_labels, original_input_size, resized_input_size, coco_image_names = data
        Evaluate_model(imgs, bboxes, labels, center_points, point_labels, original_input_size, resized_input_size, coco_image_names)
    Evaluate.postprocess()
        
if __name__ == "__main__":
    main()