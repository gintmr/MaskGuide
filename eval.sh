# export test_prompts=bbox
# python /data2/wuxinrui/RA-L/MobileSAM/distill_eval.py --val_anno_IMC /data2/wuxinrui/Datasets/IMC_1000/jsons_for_salient_instance_segmentation/test_5_prompts.json 


# export test_prompts=point
# python /data2/wuxinrui/RA-L/MobileSAM/distill_eval.py --val_anno_IMC /data2/wuxinrui/Datasets/IMC_1000/jsons_for_salient_instance_segmentation/test_5_prompts.json 


# export test_prompts=point
# python /data2/wuxinrui/RA-L/MobileSAM/distill_eval.py --val_anno_IMC /data2/wuxinrui/Datasets/IMC_1000/jsons_for_salient_instance_segmentation/test_1_prompts.json 



# export validate=v1
# python /data2/wuxinrui/RA-L/MobileSAM/distill_eval.py
# export validate=v2
# python /data2/wuxinrui/RA-L/MobileSAM/distill_eval.py
# export validate=v3
# python /data2/wuxinrui/RA-L/MobileSAM/distill_eval.py
export validate=v4
python /data2/wuxinrui/RA-L/MobileSAM/distill_eval.py
export validate=v5
python /data2/wuxinrui/RA-L/MobileSAM/distill_eval.py
export validate=v6
python /data2/wuxinrui/RA-L/MobileSAM/distill_eval.py
export validate=v7
python /data2/wuxinrui/RA-L/MobileSAM/distill_eval.py