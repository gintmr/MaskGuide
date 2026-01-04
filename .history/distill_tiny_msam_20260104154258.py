
import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1" 
import argparse

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import ConcatDataset

from DistillFinetune.Imgencoder_Distill import Imgencoder_Distill
from Datasets.coco import Coco2MaskDataset, Coco2IMGDataset, Coco2MaskDataset_repeat, val_COCODataset

from Tools_weights.trans_ckpt import trans_ckpt
from torch.utils.data import ConcatDataset, WeightedRandomSampler
import torch
import numpy as np
os.environ['NCCL_P2P_DISABLE'] = '1'

torch.set_float32_matmul_precision('high')  # 或'medium'
# torch.set_float32_matmul_precision('highest')  # 禁用Tensor Core加速

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,4'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
NUM_GPUS = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
DEVICE = 'cuda'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ['INFERENCE_MODE'] = "test"# 是否下采样1/2
# os.environ['INFERENCE_MODE'] = "train"# 是否下采样1/2
os.environ['MODEL_MODE'] = "test" # 是否构造完整模型
# os.environ['distill'] = "mask&unmask_v1"
os.environ['distill'] = "ori"
os.environ['test_prompts'] = "bbox"
# torch.cuda.set_per_process_memory_fraction(0.9, device=0)
# torch.cuda.set_per_process_memory_fraction(0.9, device=1)
# torch.cuda.set_per_process_memory_fraction(0.9, device=2)
# torch.cuda.set_per_process_memory_fraction(0.9, device=3)



def main():
    parser = argparse.ArgumentParser()

    #################G IMC
    parser.add_argument("--train_data_IMC", default="/data2/wuxinrui/HF_data/MIMC_FINAL_standard/train_list", type=str, required=False, help="path to the data root")
    parser.add_argument("--train_anno_IMC", default="/data2/wuxinrui/HF_data/MIMC_FINAL_standard/train.json", type=str, required=False, help="path to the annotation file")
    parser.add_argument("--val_data_IMC", default="/data2/wuxinrui/HF_data/MIMC_FINAL_standard/val_list", type=str, required=False, help="path to the data root")
    parser.add_argument('--val_anno_IMC', default="/data2/wuxinrui/HF_data/MIMC_FINAL_standard/val.json", )

    ################G UIIS
    parser.add_argument("--train_data_UIIS", default="/data2/wuxinrui/HF_data/UIIS/train", type=str, required=False, help="path to the data root")
    parser.add_argument("--train_anno_UIIS", default="/data2/wuxinrui/HF_data/UIIS_standard/train.json", type=str, required=False, help="path to the annotation file")
    parser.add_argument("--val_data_UIIS", default="/data2/wuxinrui/HF_data/UIIS/val", type=str, required=False, help="path to the data root")
    parser.add_argument('--val_anno_UIIS', default="/data2/wuxinrui/HF_data/UIIS_standard/val.json", )

    #############G COCO
    parser.add_argument("--train_data_COCO", default="/data2/wuxinrui/HF_data/COCO/images/train2017", type=str, required=False, help="path to the data root")
    parser.add_argument("--train_anno_COCO", default="/data2/wuxinrui/HF_data/COCO_standard/train.json", type=str, required=False, help="path to the annotation file")
    parser.add_argument("--val_data_COCO", default="/data2/wuxinrui/HF_data/COCO/images/val2017", type=str, required=False, help="path to the data root")
    parser.add_argument('--val_anno_COCO', default="/data2/wuxinrui/HF_data/COCO_standard/val.json", type=str, required=False, help="path to the annotation file")

    # #############G VOC
    parser.add_argument("--train_data_VOC", default="/data2/wuxinrui/Datasets/VOCdevkit/VOC2007/JPEGImages", type=str, required=False, help="path to the data root")
    parser.add_argument("--train_anno_VOC", default="/data2/wuxinrui/Datasets/VOCdevkit/VOC2007/COCO_Annotations/train2017.json", type=str, required=False, help="path to the annotation file")
    parser.add_argument("--val_data_VOC", default="/data2/wuxinrui/Datasets/VOCdevkit/VOC2007/JPEGImages", type=str, required=False, help="path to the data root")
    parser.add_argument('--val_anno_VOC', default="/data2/wuxinrui/Datasets/VOCdevkit/VOC2007/COCO_Annotations/val2017.json", type=str, required=False, help="path to the annotation file") 


    ###############G Target
    parser.add_argument("--Tar", default="Img_Encoder", choices=["Img_Encoder", "Mask_Decoder", "Prompt_Encoder"],type=str, required=False, help="target to be distilled")

    # parser.add_argument("--T_model", default='tiny_msam', type=str, required=False, help="model type")
    parser.add_argument("--T_model", default='vit_t', type=str, required=False, help="model type")
    # parser.add_argument("--T_model", default='vit_h', type=str, required=False, help="model type")
    # parser.add_argument("--S_model", default='vit_h', type=str, required=False, help="model type")
    parser.add_argument("--S_model", default='tiny_msam', type=str, required=False, help="model type")
    # parser.add_argument("--S_model", default='vit_t', type=str, required=False, help="model type")
    # parser.add_argument("--S_model", default='vit_h', type=str, required=False, help="model type")
    # parser.add_argument("--S_model", default='micro_sam', type=str, required=False, help="model type")
    # parser.add_argument("--T_checkpoint_path", default="/data2/wuxinrui/Distill-SAM/weights/sam_vit_h_4b8939.pth", type=str, required=False, help="path to the checkpoint") #g vit_h
    # parser.add_argument("--T_checkpoint_path", default="/data2/wuxinrui/Distill-SAM/weights/temp_weights/UIIS854IMC947.pth", type=str, required=False, help="path to the checkpoint") #g vit_h
    parser.add_argument("--T_checkpoint_path", default="/data2/wuxinrui/Distill-SAM/weights/mobile_sam.pt", type=str, required=False, help="path to the checkpoint")
    # parser.add_argument("--T_checkpoint_path", default="/data2/wuxinrui/Distill-SAM/weights/temp_weights/distillv3.pth", type=str, required=False, help="path to the checkpoint") #g vit_t
    # parser.add_argument("--T_checkpoint_path", default="/data2/wuxinrui/Distill-SAM/trained_models/Img_Encoder_T_vit_h_S_vit_t/temp/distill_v2.pth", type=str, required=False, help="path to the checkpoint")
    # parser.add_argument("--T_checkpoint_path", default="/data2/wuxinrui/Distill-SAM/trained_models/Img_Encoder_T_vit_t_S_tiny_msam/temp_copy/stroke_v8.pth", type=str, required=False, help="path to the checkpoint")

    # parser.add_argument("--S_checkpoint_path", default="/data2/wuxinrui/Distill-SAM/weights/temp_weights/UIIS854IMC947.pth", type=str, required=False, help="path to the checkpoint") #g vit_h
    # parser.add_argument("--S_checkpoint_path", default="/data2/wuxinrui/Distill-SAM/trained_models/Img_Encoder_copy_copy_T_vit_t_S_tiny_msam/temp/only_distill_50epoch_mask&unmask_8870.pth", type=str, required=False, help="path to the checkpoint") #g tiny_msam
    # parser.add_argument("--S_checkpoint_path", default="/data2/wuxinrui/Distill-SAM/weights/mobile_sam.pt", type=str, required=False, help="path to the checkpoint") #g vit_t
    # parser.add_argument("--S_checkpoint_path", default="/data2/wuxinrui/Distill-SAM/trained_models/Img_Encoder_T_vit_t_S_micro_sam/temp/micro_sam_stage11.pth", type=str, required=False, help="path to the checkpoint") #g micro_sam
    parser.add_argument("--S_checkpoint_path", default="/data2/wuxinrui/Distill-SAM/weights/init_tiny_msam.pth", type=str, required=False, help="path to the checkpoint") #g tiny_msam


    # 添加一个名为multimask的参数，类型为布尔型，默认值为False，当该参数被指定时，其值为True，用于生成多掩码
    parser.add_argument("--multimask", action="store_true", help="generate multi masks")
    # 添加一个名为use_bbox的参数，类型为布尔型，默认值为False，当该参数被指定时，其值为True，用于生成多掩码
    parser.add_argument("--use_bbox", default=False, help="generate multi masks")
    parser.add_argument("--use_centerpoint", action="store_true", help="use only one center point", default=False)
    parser.add_argument('--every_n_train_steps', default=50)
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--save_topk", type=int, default=5, help="save top K models")
    parser.add_argument("--image_size", type=int, default=1024, help="image size")
    parser.add_argument("--epochs", type=int, default=20, help="number of steps")
    parser.add_argument("--num_points", type=int, default=5, help="number of random points")
    parser.add_argument("--length", type=int, default=10, help="the length of the chosen masks")

    parser.add_argument("--learning_rate", type=float, default=5.0e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight decay")
    parser.add_argument("--metrics_interval", type=int, default=500, help="interval for logging metrics")

    parser.add_argument("--only_distill", default=True)
    parser.add_argument("--add_distill", default=True)
    parser.add_argument("--repeat_sample", default=False)

    args = parser.parse_args()

    # class combined_datasets(ConcatDataset):
    #     def __init__(self, datasets):
    #         super().__init__(datasets)
    #     @classmethod
    #     def collate_fn(cls, batch):
    #         images, bboxes, masks, center_points, point_labels, img_name, category_ids ,original_input_size, resized_input_size, coco_image_names = zip(*batch)
    #         images = torch.stack(images, dim=0)
    #         return images, bboxes, masks, center_points, point_labels, img_name, category_ids, original_input_size, resized_input_size, coco_image_names


    class combined_datasets(ConcatDataset):
        def __init__(self, datasets, sample_weights=None):
            super().__init__(datasets)
            self.datasets = datasets
            self.sample_weights = sample_weights if sample_weights is not None else [1.0 / len(datasets)] * len(datasets)
            self._build_index_map()

        def _build_index_map(self):
            """构建索引映射表，记录每个样本的原始索引和所属数据集"""
            self.index_map = []
            for dataset_idx, dataset in enumerate(self.datasets):
                for sample_idx in range(len(dataset)):
                    self.index_map.append((dataset_idx, sample_idx))

        def __getitem__(self, idx):
            """根据权重随机选择数据集和样本"""
            if self.sample_weights:
                # 根据权重选择数据集
                dataset_idx = np.random.choice(len(self.datasets), p=self.sample_weights)
                # 从选中的数据集中随机选择一个样本
                sample_idx = np.random.randint(0, len(self.datasets[dataset_idx]))
                return self.datasets[dataset_idx][sample_idx]
            else:
                # 默认均匀采样
                return super().__getitem__(idx)

        @classmethod
        def collate_fn(cls, batch):
            images, bboxes, masks, center_points, point_labels, img_name, original_input_size, resized_input_size, coco_image_names = zip(*batch)
            images = torch.stack(images, dim=0)
            return images, bboxes, masks, center_points, point_labels, img_name, original_input_size, resized_input_size, coco_image_names


    if not args.repeat_sample:
        print(f"distill dataset!!!!!!!!!!!!")
        dataset = Coco2MaskDataset
    else:
        dataset = Coco2MaskDataset_repeat
        ## 仅在使用outline增强训练时，repeat sample用于多次重复采样数据
    dataset = Coco2MaskDataset
    # load the dataset
    # train_dataset_IMC = dataset(data_root=args.train_data_IMC, annotation_path=args.train_anno_IMC, image_size=args.image_size,length=args.length,num_points=args.num_points,use_centerpoint=args.use_centerpoint)
    # train_dataset_UIIS = dataset(data_root=args.train_data_UIIS, annotation_path=args.train_anno_UIIS, image_size=args.image_size,length=args.length,num_points=args.num_points,use_centerpoint=args.use_centerpoint)
    train_dataset_COCO = dataset(data_root=args.train_data_COCO, annotation_path=args.train_anno_COCO, image_size=args.image_size,length=args.length,num_points=args.num_points,use_centerpoint=args.use_centerpoint)

    train_dataset = combined_datasets([train_dataset_COCO])
    # train_dataset = combined_datasets([train_dataset_UIIS, train_dataset_IMC], sample_weights=[0.6, 0.4])
    # train_dataset = combined_datasets([train_dataset_IMC])
    # train_dataset = combined_datasets([train_dataset_UIIS])

    val_dataset = Coco2MaskDataset(data_root=args.val_data_IMC, annotation_path=args.val_anno_IMC, image_size=args.image_size,length=args.length, num_points=args.num_points,use_centerpoint=args.use_centerpoint)
    # val_dataset = Coco2MaskDataset(data_root=args.val_data_UIIS, annotation_path=args.val_anno_UIIS, image_size=args.image_size,length=args.length, num_points=args.num_points,use_centerpoint=args.use_centerpoint)


    steps_in_epoch = len(train_dataset) // (args.batch_size * NUM_GPUS)
    max_steps = steps_in_epoch * args.epochs
    print("steps_in_epoch:", steps_in_epoch)
    print("max_steps:", max_steps)
    
    log_dir = "./metrics_logs/" + args.Tar + f"_T_{args.T_model}_S_{args.S_model}_onlydistill_{args.only_distill}_adddistill_{args.add_distill}"
    
    output_dir = os.path.join("/data2/wuxinrui/Distill-SAM/trained_models/", f"{args.Tar}_T_{args.T_model}_S_{args.S_model}_onlydistill_{args.only_distill}_adddistill_{args.add_distill}")
    
    save_model_name = args.Tar + ".pth"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

        # create the model
    if ".ckpt" in  args.S_checkpoint_path:
        temp_checkpoint_path = trans_ckpt(args.S_checkpoint_path)
    else:
        temp_checkpoint_path = args.S_checkpoint_path
    
    args.S_checkpoint_path = temp_checkpoint_path

    model = Imgencoder_Distill(
        args.T_model,
        args.S_model,
        args.T_checkpoint_path,
        args.S_checkpoint_path,
        freeze_image_encoder=False,
        freeze_prompt_encoder=True,
        freeze_mask_decoder=True,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        metrics_interval=args.metrics_interval,
        multimask=args.multimask,
        use_bbox=args.use_bbox,
        max_steps=max_steps,
        epochs=args.epochs,
        distill_weight=1, #g 初始化权重时 0.2
        only_distill=args.only_distill,
        add_distill=args.add_distill,
    )
    # 定义回调函数列表
    callbacks = [
        # 学习率监控器，每一步记录一次学习率
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            dirpath=output_dir,
            filename='{step}-{val_av_BS_IoU:.4f}',
            save_last=True,  # 保存最新的模型
            save_top_k=args.save_topk,  # 保存最好的模型数量
            monitor="val_av_BS_IoU",  # 监控的指标
            mode="max",  # 监控指标的最大值
            save_weights_only=False,  # 只保存模型权重
            every_n_train_steps=args.every_n_train_steps,  # 每训练args.metrics_interval步保存一次模型
        ),
    ]
    logger = TensorBoardLogger(save_dir=log_dir, name="SAM_DISTILL")
    trainer = pl.Trainer(
        strategy='ddp' if NUM_GPUS > 1 else None,
        accelerator=DEVICE,
        devices=NUM_GPUS,
        precision=32,
        callbacks=callbacks,
        max_epochs=-1,
        logger=logger,
        max_steps=max_steps,
        val_check_interval=1.0,
        # val_check_interval=5000,
        check_val_every_n_epoch=1,
        # check_val_every_n_epoch=None,
        limit_val_batches=1.0, 
        num_sanity_val_steps=1,
        profiler="simple",
        gradient_clip_val=1.0,
        # accumulate_grad_batches=4
    )

    # trainer.validate(model)
    trainer.fit(model, )

    # save_checkpoint = {}
    # for k, v in model.S_model.state_dict().items():
    #     if "S_model." in k:
    #         # k = k.replace("S_model.", "")
    #         save_checkpoint[k.replace("S_model.", "")] = v
        
    # teacher_checkpoint = torch.load(args.checkpoint_path)
    # for k, v in teacher_checkpoint.items():
    #     if k not in save_checkpoint:
    #         save_checkpoint[k] = v
    
    # torch.save(model, os.path.join(output_dir, save_model_name))

if __name__ == "__main__":
    main()
