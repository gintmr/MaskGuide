
import os
import numpy as np
from PIL import Image
import cv2
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import Dataset
from mobile_sam.utils.transforms import ResizeLongestSide
from eval_tools import random_croods_in_mask, random_croods_in_mask_5pieces
from pycocotools.coco import COCO
# module.py
import torchvision.transforms as T
from torchvision.transforms import ToPILImage
import random
import cv2
import numpy as np

def draw_random_contours(image, contours, draw_ratio=0.5, color=(255, 255, 255), thickness=2):
    """
    随机绘制部分轮廓
    :param image: 原始图像
    :param contours: 轮廓列表
    :param draw_ratio: 绘制的点的比例（0 到 1 之间）
    :param color: 绘制的颜色
    :param thickness: 线条的厚度
    :return: 绘制后的图像
    """
    for contour in contours:
        # 随机选择一部分点
        num_points = int(len(contour) * draw_ratio)
        random_indices = np.random.choice(len(contour), num_points, replace=False)
        random_points = contour[random_indices]

        # 将随机点转换为连续的轮廓格式
        random_contour = random_points.reshape(-1, 1, 2).astype(np.int32)

        # 绘制随机轮廓
        cv2.polylines(image, [random_contour], isClosed=False, color=color, thickness=thickness)

    return image


class val_COCODataset(Dataset):
    def __init__(self, data_root, image_size=None, annotation_path=None, length=150, num_points=5, use_centerpoint=False):
        self.coco = COCO(annotation_path)  # 使用 pycocotools 加载 COCO 数据集
        self.data_root = data_root
        assert image_size is not None, "image_size must be specified"
        self.image_size = image_size
        self.transform = ResizeLongestSide(self.image_size) #自定义最长边
        self.length = length #G the number of masks to load
        self.num_points = num_points
        self.use_centerpoint = use_centerpoint
        self.imgIds = self.coco.getImgIds()[:]
        self.distill_target = os.getenv("distill", "ori")

        
    def preview_input(self, img, coco_image_name):
        import torchvision.transforms.functional as F
        preprocessed_image_pil = F.to_pil_image(img)
        save_dir = "/data2/wuxinrui/Distill-SAM/preview_data"
        save_path = os.path.join(save_dir, coco_image_name)
        os.makedirs(save_dir, exist_ok=True)
        preprocessed_image_pil.save(save_path)
        print(f"Processed {coco_image_name} saved to {save_path}")
        
    def preprocess(self, x):
        """Normalize pixel values and pad to a square input."""
        ## 将resize后的图像padding至image_encoder接受的长度
        # Normalize colors

        h, w = x.shape[:2]
        padh = self.image_size - h
        padw = self.image_size - w

        x = np.pad(x, ((0, padh), (0, padw), (0, 0)), mode='constant', constant_values=0)  # 假设 x 是 (H, W, C) 格式
        return x
    
    def preprocess_mask(self, x):
        """Normalize pixel values and pad to a square input."""
        ## 将resize后的图像padding至image_encoder接受的长度
        # Normalize colors

        h, w = x.shape[:2]  # 获取图像的高度和宽度
        padh = self.image_size - h  # 计算需要填充的高度
        padw = self.image_size - w  # 计算需要填充的宽度

        x = np.pad(x, ((0, padh), (0, padw)), mode='constant', constant_values=0)  # 假设 x 是 (H, W, C) 格式
        return x
        
    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, index):
        try:
            # 使用 pycocotools 获取图像信息
            img_id = self.imgIds[index]
            img_info = self.coco.loadImgs(img_id)[0]
            coco_image_name = img_info["file_name"]
            image_path = os.path.join(self.data_root, coco_image_name)
            image = Image.open(image_path).convert("RGB")
            image = np.array(image)

            original_height, original_width = image.shape[0], image.shape[1]
            
            input_image = self.transform.apply_image(image) ## 根据设置的最长边resize图像
            
            
            resized_height, resized_width = input_image.shape[0], input_image.shape[1]

            original_input_size = [original_height, original_width] 
            resized_input_size = [resized_height, resized_width] 
            annIds = self.coco.getAnnIds(imgIds=img_id)
            annotations = self.coco.loadAnns(annIds)

            bboxes = []
            masks = []
            center_point_labels = []
            center_points = []
            combined_points = []
            combined_point_labels = []
            coords_list = []
            coords_labels = []
            category_ids = []
            
            input_image = self.preprocess(input_image)
            masked_image = np.zeros_like(input_image)
            
            for annotation in annotations[:self.length]:
                x, y, w, h = annotation["bbox"]
                if self.distill_target == "mask":
                    coords = torch.zeros((0, 2))
                else:
                    coords = torch.from_numpy(np.array(annotation['coords']))

                bbox = [x, y, x + w, y + h]

                # resize掩码
                mask = self.coco.annToMask(annotation)
                ## 保持原来的尺寸
                category_ids.append([annotation["category_id"]])
                # points, num_points = random_croods_in_mask(mask=mask, num_croods=self.num_points) ## points的坐标顺序与mask相同,W\H
                # points, num_points = random_croods_in_mask_5pieces(mask=mask) ## points的坐标顺序与mask相同,W\H
                #g 分成5个区域，每个区域选两点
                
                mask = cv2.resize(mask, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
                mask = (mask > 0.5).astype(np.uint8)
                mask = self.preprocess_mask(mask)
                # points = self.transform.apply_coords_torch(points, original_input_size)
                # bbox = self.transform.apply_boxes_torch(bbox, original_input_size)

                # 将mask应用到图像上
                masked_region = cv2.bitwise_and(input_image, input_image, mask=mask)
                # 将处理后的区域叠加到全黑图像上
                masked_image = cv2.add(masked_image, masked_region)

                ## bbox需要resize,同理center_points需要resize, points也需要resize; 但是mask不resize

                center_points.append([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
                bboxes.append(bbox)
                masks.append(mask)
                coords = self.transform.apply_coords_torch(coords, original_input_size)
                #G assert mask has the same size as original_input_size
                # assert mask.shape == (original_input_size[0], original_input_size[1])
                coords_list.append(coords)
                coords_labels.append([1]*len(coords))
                # combined_point_labels.append([count_label] * num_points)
                # combined_points.append(points)
                # center_point_labels.append([count_label])

            # coords_list = self.transform.apply_coords_torch(coords_list, original_input_size)
            # center_points = self.transform.apply_coords_torch(center_points, original_input_size)
            bboxes = self.transform.apply_boxes_torch(bboxes, original_input_size)
            
            # center_points = np.stack(center_points, axis=0)
            coords_labels = np.stack(coords_labels, axis=0)
            coords_list = np.stack(coords_list, axis=0)

            category_ids = np.array([0])  #g 用不上

            bboxes = np.stack(bboxes, axis=0)
            
            masks = np.stack(masks, axis=0) 
            # point_labels = np.stack(point_labels, axis=0)

            # 将输入图像转换为torch张量
            # input_image = self.preprocess(input_image)
            # self.preview_input(original_image_with_boundaries, coco_image_name) #g 预览输入图像
            input_image_torch = torch.tensor(input_image)
            #g 将张量的维度从HWC转换为CHW
            input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()
            
            # masked_image = self.preprocess(masked_image)
            masked_image_torch = torch.tensor(masked_image)
            masked_image_torch = masked_image_torch.permute(2, 0, 1).contiguous()


            combined_image = torch.cat([input_image_torch, masked_image_torch], dim=0)  # 结果形状为[6, H, W]
            # print(f"input_image_torch shape: {input_image_torch.shape}")
            # print(f"masked_image_torch shape: {masked_image_torch.shape}")
            if self.distill_target == "mask":
                return_img = combined_image
            elif self.distill_target == "ori":
                return_img = input_image_torch

            return (
                return_img,
                torch.tensor(bboxes),
                torch.tensor(masks).long(),
                # torch.tensor(0.0),
                # torch.tensor(0.0),
                torch.tensor(coords_list),
                torch.tensor(coords_labels),
                coco_image_name,
                torch.tensor(category_ids),
                original_input_size,
                resized_input_size,
                str(coco_image_name),
            )
        except Exception as e:
            import traceback
            print("Error in loading image: ", coco_image_name)
            # print(coco_image_name)
            print("Error: ", e)
            # raise
            # error_message = traceback.format_exc()
            # print("Detailed Error: ", error_message)
            return self.__getitem__((index+1) % len(self))
        
        
    @classmethod
    def collate_fn(cls, batch):
        images, bboxes, masks, center_points, point_labels, img_name, category_ids ,original_input_size, resized_input_size, coco_image_names = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, bboxes, masks, center_points, point_labels, img_name, category_ids, original_input_size, resized_input_size, coco_image_names


    



class Coco2MaskDataset(Dataset):
    def __init__(self, data_root, image_size=None, annotation_path=None, length=150, num_points=5, use_centerpoint=False):
        self.coco = COCO(annotation_path)  # 使用 pycocotools 加载 COCO 数据集
        self.data_root = data_root
        assert image_size is not None, "image_size must be specified"
        self.image_size = image_size
        self.transform = ResizeLongestSide(self.image_size) #自定义最长边
        self.length = length #G the number of masks to load
        self.num_points = num_points
        self.use_centerpoint = use_centerpoint
        self.imgIds = self.coco.getImgIds()[:]
        
        self.augmentations = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机调整亮度、对比度、饱和度和色调
            T.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)), # 高斯模糊
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.5), # 随机调整锐度
            T.ToTensor(),  # 将 PIL 图像转换为张量
            # T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.5, inplace=False),  # 随机遮挡
            ToPILImage(),  # 将张量转换回 PIL 图像
        ])
        self.distill_target = os.getenv("distill", "ori")
        print(f"distill_target: {self.distill_target}\n" * 10)
        
    def preview_input(self, img, coco_image_name):
        import torchvision.transforms.functional as F
        preprocessed_image_pil = F.to_pil_image(img)
        save_dir = "/data2/wuxinrui/Distill-SAM/preview_data"
        save_path = os.path.join(save_dir, coco_image_name)
        os.makedirs(save_dir, exist_ok=True)
        preprocessed_image_pil.save(save_path)
        print(f"Processed {coco_image_name} saved to {save_path}")
        
    def preprocess(self, x):
        """Normalize pixel values and pad to a square input."""
        ## 将resize后的图像padding至image_encoder接受的长度
        # Normalize colors

        h, w = x.shape[:2]
        padh = self.image_size - h
        padw = self.image_size - w

        x = np.pad(x, ((0, padh), (0, padw), (0, 0)), mode='constant', constant_values=0)  # 假设 x 是 (H, W, C) 格式
        return x
    
    def preprocess_mask(self, x):
        """Normalize pixel values and pad to a square input."""
        ## 将resize后的图像padding至image_encoder接受的长度
        # Normalize colors

        h, w = x.shape[:2]  # 获取图像的高度和宽度
        padh = self.image_size - h  # 计算需要填充的高度
        padw = self.image_size - w  # 计算需要填充的宽度

        x = np.pad(x, ((0, padh), (0, padw)), mode='constant', constant_values=0)  # 假设 x 是 (H, W, C) 格式
        return x
        
    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, index):
        try:
            # 使用 pycocotools 获取图像信息
            img_id = self.imgIds[index]
            img_info = self.coco.loadImgs(img_id)[0]
            coco_image_name = img_info["file_name"]
            image_path = os.path.join(self.data_root, coco_image_name)
            image = Image.open(image_path).convert("RGB")
            # image = self.augmentations(image) if random.random() > 0.3 else image
            # image = self.augmentations(image)
            image = np.array(image)

            # if image.shape[-1] == 3:  # HWC格式
            #     image = image.transpose(2, 0, 1)  # 转为CHW
            # elif image.shape[0] == 3:  # CHW格式
            #     pass

            original_height, original_width = image.shape[0], image.shape[1]
            
            input_image = self.transform.apply_image(image) ## 根据设置的最长边resize图像
            
            
            resized_height, resized_width = input_image.shape[0], input_image.shape[1]

            original_input_size = [original_height, original_width] 
            resized_input_size = [resized_height, resized_width] 
            annIds = self.coco.getAnnIds(imgIds=img_id)
            annotations = self.coco.loadAnns(annIds)

            bboxes = []
            masks = []
            center_point_labels = []
            center_points = []
            combined_points = []
            combined_point_labels = []
            category_ids = []
            count_label = 1

            
            #g test-train模式混合训练
            # all_masks_num = len(annotations)
            # if all_masks_num >= 50:
            #     os.environ['INFERENCE_MODE'] = "train"
            # else:
            #     rate = random.uniform(0, 1)
            #     os.environ['INFERENCE_MODE'] = 'test' if rate <= 0.5 else 'train'
            rate = random.uniform(0, 1)
            # os.environ['INFERENCE_MODE'] = 'test' if rate <= 0.25 else 'train'
            #g test-train模式混合训练

            input_image = self.preprocess(input_image)
            masked_image = np.zeros_like(input_image)
            original_image_with_boundaries = input_image.copy()
            unmasked_image = input_image.copy()

            for annotation in annotations[:self.length]:
                x, y, w, h = annotation["bbox"]

                bbox = [x, y, x + w, y + h]

                # resize掩码
                mask = self.coco.annToMask(annotation)
                ## 保持原来的尺寸
                category_ids.append([annotation["category_id"]])
                # points, num_points = random_croods_in_mask(mask=mask, num_croods=self.num_points) ## points的坐标顺序与mask相同,W\H
                if 'coords' in annotation:
                    points = annotation['coords']
                    num_points = len(points)
                    # print("validating !!!")
                    prompt_type = "all_points"
                else:
                    points, num_points = random_croods_in_mask_5pieces(mask=mask, points_per_segment=6)
                    weights = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                    prompt_types = ['1_point', '3_points', '5_points', '8_points', '10_points', "all_points"]
                    prompt_type = random.choices(prompt_types, weights=weights, k=1)[0]

                # points, num_points = random_croods_in_mask_5pieces(mask=mask) ## points的坐标顺序与mask相同,W\H
                #g 分成5个区域，每个区域选两点
                
                mask = cv2.resize(mask, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
                mask = (mask > 0.5).astype(np.uint8)
                mask = self.preprocess_mask(mask)
                # points = self.transform.apply_coords_torch(points, original_input_size)
                # bbox = self.transform.apply_boxes_torch(bbox, original_input_size)
                
                #! 将mask应用到图像上
                masked_region = cv2.bitwise_and(input_image, input_image, mask=mask)
                # 将处理后的区域叠加到全黑图像上
                masked_image = cv2.add(masked_image, masked_region)
                
                #! 将mask反向应用到图像上
                # inverse_mask = cv2.bitwise_not(mask)
                # # unmasked_image = cv2.bitwise_and(input_image, input_image, mask=inverse_mask)
                # unmasked_region = cv2.bitwise_and(input_image, input_image, mask=inverse_mask)
                # unmasked_image = cv2.add(unmasked_image, unmasked_region)
                unmasked_image[mask > 0] = [0, 0, 0]

                #! 将outline画到图像上
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(original_image_with_boundaries, contours, -1, (255, 255, 255), 1)


                ## bbox需要resize,同理center_points需要resize, points也需要resize; 但是mask不resize

                center_points.append([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
                bboxes.append(bbox)
                masks.append(mask)
                
                #G assert mask has the same size as original_input_size
                # assert mask.shape == (original_input_size[0], original_input_size[1])
                combined_point_labels.append([count_label] * num_points)
                combined_points.append(points)
                center_point_labels.append([count_label])

            img_extname = os.path.splitext(coco_image_name)[1]
            
            # self.preview_input(masked_image, coco_image_name.replace(img_extname, "_masked" + img_extname))
            # self.preview_input(unmasked_image, coco_image_name.replace(img_extname, "_unmasked" + img_extname))
            # self.preview_input(original_image_with_boundaries, coco_image_name.replace(img_extname, "_outline" + img_extname))
            # self.preview_input(input_image, coco_image_name.replace(img_extname, "_original" + img_extname))
            
            combined_points = self.transform.apply_coords_torch(combined_points, original_input_size)
            center_points = self.transform.apply_coords_torch(center_points, original_input_size)
            bboxes = self.transform.apply_boxes_torch(bboxes, original_input_size)
            
            center_points = np.stack(center_points, axis=0)
            combined_points = np.stack(combined_points, axis=0)
            # category_ids = np.stack(category_ids, axis=0)

            if self.use_centerpoint:
                given_points = center_points
                point_labels = center_point_labels
            else:
                given_points = combined_points
                point_labels = combined_point_labels

            bboxes = np.stack(bboxes, axis=0)
            
            masks = np.stack(masks, axis=0) 
            point_labels = np.stack(point_labels, axis=0)

            # 将输入图像转换为torch张量


            if prompt_type == "1_point":
                given_points = given_points[:, :1]
                point_labels = point_labels[:, :1]
            elif prompt_type == "3_points":
                given_points = given_points[:, :3]
                point_labels = point_labels[:, :3]
            elif prompt_type == "5_points":
                    # Select points with indices 0, 2, 4, 6, 8
                selected_indices = [0, 2, 4, 6, 8]
                given_points = given_points[:, selected_indices]
                point_labels = point_labels[:, selected_indices]
                #g 用于匹配分5块选取的points
            elif prompt_type == "8_points":
                given_points = given_points[:, :8]
                point_labels = point_labels[:, :8]
            elif prompt_type == "10_points":
                given_points = given_points[:, :10]
                point_labels = point_labels[:, :10]
            elif prompt_type == "all_points":
                given_points = given_points
                point_labels = point_labels

            input_image_torch = torch.tensor(input_image)
            #g 将张量的维度从HWC转换为CHW
            input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()
            # masked_image = self.preprocess(masked_image)
            masked_image_torch = torch.tensor(masked_image)
            masked_image_torch = masked_image_torch.permute(2, 0, 1).contiguous()
            unmasked_image_torch = torch.tensor(unmasked_image)
            unmasked_image_torch = unmasked_image_torch.permute(2, 0, 1).contiguous()
            original_image_with_boundaries_torch = torch.tensor(original_image_with_boundaries)
            original_image_with_boundaries_torch = original_image_with_boundaries_torch.permute(2, 0, 1).contiguous()

            if self.distill_target == "mask":
                return_img = torch.cat([input_image_torch, masked_image_torch], dim=0)
            elif self.distill_target == "unmask":
                return_img = torch.cat([input_image_torch, unmasked_image_torch], dim=0)
            elif "mask&unmask" in self.distill_target:
                return_img = torch.cat([input_image_torch, masked_image_torch, unmasked_image_torch], dim=0)
            elif self.distill_target == "ori":
                return_img = input_image_torch
            elif self.distill_target == "outline":
                return_img = torch.cat([input_image_torch, original_image_with_boundaries_torch], dim=0)
            elif self.distill_target == "mask&outline":
                return_img = torch.cat([input_image_torch, masked_image_torch, original_image_with_boundaries_torch], dim=0)
            
            # print(f"return_img.shape = {return_img.shape}")

            return (
                return_img,
                torch.tensor(bboxes),
                torch.tensor(masks).long(),
                torch.tensor(given_points),
                torch.tensor(point_labels),
                coco_image_name,
                # torch.tensor(category_ids),
                original_input_size,
                resized_input_size,
                str(coco_image_name),
            )
        except Exception as e:
            import traceback
            print("Error in loading image: ", coco_image_name)
            print("Error: ", e)
            # raise
            # error_message = traceback.format_exc()
            # print("Detailed Error: ", error_message)
            return self.__getitem__((index+1) % len(self))
        
        
    @classmethod
    def collate_fn(cls, batch):
        images, bboxes, masks, center_points, point_labels, img_name ,original_input_size, resized_input_size, coco_image_names = zip(*batch)
        images = torch.stack(images, dim=0)
        # print(f"images.shape = {images.shape}")
        return images, bboxes, masks, center_points, point_labels, img_name, original_input_size, resized_input_size, coco_image_names



class Coco2IMGDataset(Dataset):
    def __init__(self, data_root, image_size=None, annotation_path=None, length=150, num_points=5, use_centerpoint=False):
        self.data_root = data_root
        self.img_name_list = os.listdir(data_root)

        assert image_size is not None, "image_size must be specified"
        self.image_size = image_size
        self.transform = ResizeLongestSide(self.image_size) #自定义最长边

        self.augmentations = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机调整亮度、对比度、饱和度和色调
            T.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)), # 高斯模糊
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.5), # 随机调整锐度
            T.ToTensor(),  # 将 PIL 图像转换为张量
            T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.5, inplace=False),  # 随机遮挡
            ToPILImage(),  # 将张量转换回 PIL 图像
        ])

    def preprocess(self, x):
        """Normalize pixel values and pad to a square input."""
        ## 将resize后的图像padding至image_encoder接受的长度
        # Normalize colors

        h, w = x.shape[:2]
        padh = self.image_size - h
        padw = self.image_size - w

        x = np.pad(x, ((0, padh), (0, padw), (0, 0)), mode='constant', constant_values=0)  # 假设 x 是 (H, W, C) 格式
        return x

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, index):
        try:
            # 使用 pycocotools 获取图像信息
            img_name = self.img_name_list[index]
            img_file_path = os.path.join(self.data_root, img_name)

            image = Image.open(img_file_path).convert("RGB")
            image = self.augmentations(image) if random.random() > 0.2 else image
            image = np.array(image)

            original_height, original_width = image.shape[0], image.shape[1]
            
            input_image = self.transform.apply_image(image) ## 根据设置的最长边resize图像
            
            
            resized_height, resized_width = input_image.shape[0], input_image.shape[1]

            original_input_size = [original_height, original_width] 
            resized_input_size = [resized_height, resized_width] 

            # 将输入图像转换为torch张量
            input_image = self.preprocess(input_image)
            input_image_torch = torch.tensor(input_image)
            
            #g 将张量的维度从HWC转换为CHW
            input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()
            
            # 手动设置未加载的参数为零张量列表
            bboxes = torch.zeros((0, 4))  # 假设 bboxes 是 [N, 4] 的张量
            masks = torch.zeros((0, self.image_size, self.image_size), dtype=torch.long)  # 假设 masks 是 [N, H, W] 的张量
            given_points = torch.zeros((0, 2))  # 假设 given_points 是 [N, 2] 的张量
            point_labels = torch.zeros((0,), dtype=torch.long)  # 假设 point_labels 是 [N] 的张量
            category_ids = torch.zeros((0,), dtype=torch.long)  # 假设 category_ids 是 [N] 的张量

            return (
                input_image_torch,
                torch.tensor(bboxes),
                torch.tensor(masks).long(),
                torch.tensor(given_points),
                torch.tensor(point_labels),
                img_name,
                # category_ids.clone().detach(),
                # torch.tensor(category_ids),
                original_input_size,
                resized_input_size,
                str(img_name),
            )

        except Exception as e:
            import traceback
            print("Error in loading image: ", img_name)
            print("Error: ", e)
            # raise
            # error_message = traceback.format_exc()
            # print("Detailed Error: ", error_message)
            return self.__getitem__((index+1) % len(self))

    @classmethod
    def collate_fn(cls, batch):
        images, bboxes, masks, center_points, point_labels, img_name ,original_input_size, resized_input_size, coco_image_names = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, bboxes, masks, center_points, point_labels, img_name, original_input_size, resized_input_size, coco_image_names




class Coco2MaskDataset_repeat(Dataset):
    def __init__(self, data_root, image_size=None, annotation_path=None, length=150, num_points=5, use_centerpoint=False):
        self.coco = COCO(annotation_path)  # 使用 pycocotools 加载 COCO 数据集
        self.data_root = data_root
        assert image_size is not None, "image_size must be specified"
        self.image_size = image_size
        self.transform = ResizeLongestSide(self.image_size) #自定义最长边
        self.length = length #G the number of masks to load
        self.num_points = num_points
        self.use_centerpoint = use_centerpoint
        self.imgIds = self.coco.getImgIds()[:]
        self.repeat_times = 3
        self.repeat_count = 0
        self.current_index = 0
        
        self.augmentations = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机调整亮度、对比度、饱和度和色调
            T.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)), # 高斯模糊
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.5), # 随机调整锐度
            T.ToTensor(),  # 将 PIL 图像转换为张量
            # T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.5, inplace=False),  # 随机遮挡
            ToPILImage(),  # 将张量转换回 PIL 图像
        ])
        
        self.valid_indices = []
        self._build_index_mapping()
        
    def _build_index_mapping(self):
        for idx, img_id in enumerate(self.imgIds):
            self.valid_indices.append(img_id)
        
    def preview_input(self, img, coco_image_name):
        import torchvision.transforms.functional as F
        preprocessed_image_pil = F.to_pil_image(img)
        save_dir = "/data2/wuxinrui/Distill-SAM/preview_data"
        save_path = os.path.join(save_dir, coco_image_name)
        os.makedirs(save_dir, exist_ok=True)
        preprocessed_image_pil.save(save_path)
        print(f"Processed {coco_image_name} saved to {save_path}")
        
    def preprocess(self, x):
        """Normalize pixel values and pad to a square input."""
        ## 将resize后的图像padding至image_encoder接受的长度
        # Normalize colors

        h, w = x.shape[:2]
        padh = self.image_size - h
        padw = self.image_size - w

        x = np.pad(x, ((0, padh), (0, padw), (0, 0)), mode='constant', constant_values=0)  # 假设 x 是 (H, W, C) 格式
        return x
    
    def preprocess_mask(self, x):
        """Normalize pixel values and pad to a square input."""
        ## 将resize后的图像padding至image_encoder接受的长度
        # Normalize colors

        h, w = x.shape[:2]  # 获取图像的高度和宽度
        padh = self.image_size - h  # 计算需要填充的高度
        padw = self.image_size - w  # 计算需要填充的宽度

        x = np.pad(x, ((0, padh), (0, padw)), mode='constant', constant_values=0)  # 假设 x 是 (H, W, C) 格式
        return x
        
    def __len__(self):
        return len(self.imgIds) * self.repeat_times

    def __getitem__(self, index):
        try:
            if self.repeat_count >= self.repeat_times:
                self.current_index = (self.current_index + 1) % len(self.imgIds)
                self.repeat_count = 0            
            # 使用 pycocotools 获取图像信息
            img_id = self.valid_indices[self.current_index]
            # img_id = self.imgIds[self.current_index]
            self.repeat_count += 1
            img_info = self.coco.loadImgs(img_id)[0]
            coco_image_name = img_info["file_name"]
            image_path = os.path.join(self.data_root, coco_image_name)
            image = Image.open(image_path).convert("RGB")
            image = self.augmentations(image) if random.random() > 0.2 else image
            # image = self.augmentations(image)
            image = np.array(image)

            original_height, original_width = image.shape[0], image.shape[1]
            
            input_image = self.transform.apply_image(image) ## 根据设置的最长边resize图像
            
            
            resized_height, resized_width = input_image.shape[0], input_image.shape[1]

            original_input_size = [original_height, original_width] 
            resized_input_size = [resized_height, resized_width] 
            annIds = self.coco.getAnnIds(imgIds=img_id)
            annotations = self.coco.loadAnns(annIds)

            bboxes = []
            masks = []
            center_point_labels = []
            center_points = []
            combined_points = []
            combined_point_labels = []
            category_ids = []
            count_label = 1

            
            #g test-train模式混合训练
            # all_masks_num = len(annotations)
            # if all_masks_num >= 50:
            #     os.environ['INFERENCE_MODE'] = "train"
            # else:
            #     rate = random.uniform(0, 1)
            #     os.environ['INFERENCE_MODE'] = 'test' if rate <= 0.5 else 'train'
            rate = random.uniform(0, 1)
            # os.environ['INFERENCE_MODE'] = 'test' if rate <= 0.25 else 'train'
            #g test-train模式混合训练

            for annotation in annotations[:self.length]:
                x, y, w, h = annotation["bbox"]

                bbox = [x, y, x + w, y + h]

                # resize掩码
                mask = self.coco.annToMask(annotation)
                ## 保持原来的尺寸
                category_ids.append([annotation["category_id"]])
                # points, num_points = random_croods_in_mask(mask=mask, num_croods=self.num_points) ## points的坐标顺序与mask相同,W\H
                points, num_points = random_croods_in_mask_5pieces(mask=mask) 
                #g 分成5个区域，每个区域选两点
                
                mask = cv2.resize(mask, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
                mask = (mask > 0.5).astype(np.uint8)
                mask = self.preprocess_mask(mask)
                # points = self.transform.apply_coords_torch(points, original_input_size)
                # bbox = self.transform.apply_boxes_torch(bbox, original_input_size)
                
                ## bbox需要resize,同理center_points需要resize, points也需要resize; 但是mask不resize
                
                
                center_points.append([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
                bboxes.append(bbox)
                masks.append(mask)
                
                #G assert mask has the same size as original_input_size
                # assert mask.shape == (original_input_size[0], original_input_size[1])
                combined_point_labels.append([count_label] * num_points)
                combined_points.append(points)
                center_point_labels.append([count_label])
                
                original_image_with_boundaries = input_image.copy()
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                r = random.random()
                RATE = float(os.getenv('RATE', "0"))
                # if self.repeat_count == 0:
                if self.repeat_count == 0 and RATE <= 0.5:
                    # original_image_with_boundaries = draw_random_contours(image=original_image_with_boundaries, contours=contours, draw_ratio=0.8, color=(0,0,0), thickness=1)
                    cv2.drawContours(original_image_with_boundaries, contours, -1, (255, 255, 255), 1)
                elif self.repeat_count == 0 and RATE > 0.5:
                    original_image_with_boundaries = original_image_with_boundaries
                #     original_image_with_boundaries = draw_random_contours(image=original_image_with_boundaries, contours=contours, draw_ratio=0.6, color=(255,255,255), thickness=1)
                    # cv2.drawContours(original_image_with_boundaries, contours, -1, (255, 255, 255), 1)
                elif self.repeat_count == 1:
                # elif self.repeat_count == 1 and RATE <= 0.5:
                    original_image_with_boundaries = original_image_with_boundaries
                    # original_image_with_boundaries = draw_random_contours(image=original_image_with_boundaries, contours=contours, draw_ratio=0.3, color=(0,0,0), thickness=1)
                    # cv2.drawContours(original_image_with_boundaries, contours, -1, (255, 255, 255), 1)
                # elif self.repeat_count == 1 and RATE > 0.5:
                #     original_image_with_boundaries = draw_random_contours(image=original_image_with_boundaries, contours=contours, draw_ratio=0.1, color=(255,255,255), thickness=1)
                elif self.repeat_count == 2:
                    original_image_with_boundaries = original_image_with_boundaries
                    # cv2.drawContours(original_image_with_boundaries, contours, -1, (255, 255, 255), 2)
                elif self.repeat_count == 3:
                    original_image_with_boundaries = original_image_with_boundaries

            combined_points = self.transform.apply_coords_torch(combined_points, original_input_size)
            center_points = self.transform.apply_coords_torch(center_points, original_input_size)
            bboxes = self.transform.apply_boxes_torch(bboxes, original_input_size)
            
            center_points = np.stack(center_points, axis=0)
            combined_points = np.stack(combined_points, axis=0)
            category_ids = np.stack(category_ids, axis=0)

            if self.use_centerpoint:
                given_points = center_points
                point_labels = center_point_labels
            else:
                given_points = combined_points
                point_labels = combined_point_labels

            bboxes = np.stack(bboxes, axis=0)
            
            masks = np.stack(masks, axis=0) 
            point_labels = np.stack(point_labels, axis=0)

            # 将输入图像转换为torch张量
            # original_image_with_boundaries = self.preprocess(original_image_with_boundaries)
            input_image = original_image_with_boundaries
            input_image = self.preprocess(input_image)
            # self.preview_input(original_image_with_boundaries, coco_image_name) #g 预览输入图像
            
            input_image_torch = torch.tensor(input_image)
            
            #g 将张量的维度从HWC转换为CHW
            input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()
            
            weights = [0.0, 0.0, 0.5, 0.0, 0.5]
            prompt_types = ['1_point', '3_points', '5_points', '8_points', '10_points']
            prompt_type = random.choices(prompt_types, weights=weights, k=1)[0]
            
            if prompt_type == "1_point":
                given_points = given_points[:, :1]
                point_labels = point_labels[:, :1]
            elif prompt_type == "3_points":
                given_points = given_points[:, :3]
                point_labels = point_labels[:, :3]
            elif prompt_type == "5_points":
                 # Select points with indices 0, 2, 4, 6, 8
                selected_indices = [0, 2, 4, 6, 8]
                given_points = given_points[:, selected_indices]
                point_labels = point_labels[:, selected_indices]
                #g 用于匹配分5块选取的points
            elif prompt_type == "8_points":
                given_points = given_points[:, :8]
                point_labels = point_labels[:, :8]
            elif prompt_type == "10_points":
                given_points = given_points[:, :10]
                point_labels = point_labels[:, :10]



            return (
                input_image_torch,
                torch.tensor(bboxes),
                torch.tensor(masks).long(),
                torch.tensor(given_points),
                torch.tensor(point_labels),
                coco_image_name,
                # torch.tensor(category_ids),
                category_ids.clone().detach(),
                original_input_size,
                resized_input_size,
                str(coco_image_name),
            )
        except Exception as e:
            import traceback
            print("Error in loading image: ", coco_image_name)
            print("Error: ", e)
            # raise
            # error_message = traceback.format_exc()
            # print("Detailed Error: ", error_message)
            return self.__getitem__((index+1) % len(self))
        
        
    @classmethod
    def collate_fn(cls, batch):
        images, bboxes, masks, center_points, point_labels, img_name, category_ids ,original_input_size, resized_input_size, coco_image_names = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, bboxes, masks, center_points, point_labels, img_name, category_ids, original_input_size, resized_input_size, coco_image_names

