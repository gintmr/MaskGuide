# #! 👇全都加上下采样


# export INFERENCE_MODE=train
# #g 原生SAM 1,3,5,bbox
# python /data2/wuxinrui/Distill-SAM/eval.py --checkpoint_path /data2/wuxinrui/Distill-SAM/weights/sam_vit_h_4b8939.pth --ori_SAM True --label_num 1 --visualize_mask_path /data2/wuxinrui/Distill-SAM/baseline_eval/ori_SAM

# python /data2/wuxinrui/Distill-SAM/eval.py --checkpoint_path /data2/wuxinrui/Distill-SAM/weights/sam_vit_h_4b8939.pth --ori_SAM True --label_num 3 --visualize_mask_path /data2/wuxinrui/Distill-SAM/baseline_eval/ori_SAM

# python /data2/wuxinrui/Distill-SAM/eval.py --checkpoint_path /data2/wuxinrui/Distill-SAM/weights/sam_vit_h_4b8939.pth --ori_SAM True --label_num 5 --visualize_mask_path /data2/wuxinrui/Distill-SAM/baseline_eval/ori_SAM



# export INFERENCE_MODE=train
# #g MObileSAM 1,3,5,bbox
# python /data2/wuxinrui/Distill-SAM/eval.py --checkpoint_path /data2/wuxinrui/Distill-SAM/weights/mobile_sam.pt --label_num 1 --visualize_mask_path /data2/wuxinrui/Distill-SAM/baseline_eval/mobilesam

# python /data2/wuxinrui/Distill-SAM/eval.py --checkpoint_path /data2/wuxinrui/Distill-SAM/weights/mobile_sam.pt --label_num 3 --visualize_mask_path /data2/wuxinrui/Distill-SAM/baseline_eval/mobilesam

# python /data2/wuxinrui/Distill-SAM/eval.py --checkpoint_path /data2/wuxinrui/Distill-SAM/weights/mobile_sam.pt --label_num 5 --visualize_mask_path /data2/wuxinrui/Distill-SAM/baseline_eval/mobilesam


export CUDA_VISIBLE_DEVICES=1
export INFERENCE_MODE=train
#g modified_mobilesam
python /data2/wuxinrui/Distill-SAM/eval.py --checkpoint_path /data2/wuxinrui/Distill-SAM/trained_models/Distilled_encoder/COCO_MIMC_plus_new26epoch.pth --label_num 1 --visualize_mask_path /data2/wuxinrui/Distill-SAM/baseline_eval/temp_

python /data2/wuxinrui/Distill-SAM/eval.py --checkpoint_path /data2/wuxinrui/Distill-SAM/trained_models/Distilled_encoder/COCO_MIMC_plus_new26epoch.pth --label_num 3 --visualize_mask_path /data2/wuxinrui/Distill-SAM/baseline_eval/temp_

python /data2/wuxinrui/Distill-SAM/eval.py --checkpoint_path /data2/wuxinrui/Distill-SAM/trained_models/Distilled_encoder/COCO_MIMC_plus_new26epoch.pth --label_num 5 --visualize_mask_path /data2/wuxinrui/Distill-SAM/baseline_eval/temp_

