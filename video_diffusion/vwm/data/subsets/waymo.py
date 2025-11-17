import os
from .common import BaseDataset
from PIL import Image
import torch
from torchvision import transforms

class WaymoDataset(BaseDataset):
    def __init__(
            self, 
            data_root="/lpai/volumes/jointmodel/yanyunzhi/data/waymo", 
            anno_file=None, 
            postfix=None,
            target_height=320, target_width=576, num_frames=25, 
            split="train"
        ):
        
        if split == "train":
            anno_file = os.path.join(data_root, "meta_info_train.json")
        elif split == "val":
            anno_file = os.path.join(data_root, "meta_info_val.json")
        else:
            anno_file = os.path.join(data_root, "meta_info_val.json")
        
        if postfix is not None:
            anno_file = anno_file.replace(".json", f"_{postfix}.json")

        if not os.path.exists(data_root):
            raise ValueError("Cannot find dataset {}".format(data_root))
        if not os.path.exists(anno_file):
            raise ValueError("Cannot find annotation {}".format(anno_file))
        
        super().__init__(data_root, anno_file, target_height, target_width, num_frames)
        print("Waymo loaded:", len(self))
        
        self.guide_preprocessor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0)
        ])
        
        self.img_preprocessor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0)
        ])
        
        self.default_processor = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def get_image_path(self, sample_dict, current_index):
        return os.path.join(self.data_root, sample_dict["frames"][current_index])
    
    def get_guidance_path(self, sample_dict, current_index):
        return os.path.join(self.data_root, sample_dict["guidances"][current_index])
    
    def get_guidance_mask_path(self, sample_dict, current_index):
        return os.path.join(self.data_root, sample_dict["guidances_mask"][current_index])
    
    def preprocess_image(self, image_path, preprocessor): # type: ignore
        image = Image.open(image_path)
        ori_w, ori_h = image.size
        if ori_w / ori_h > self.target_width / self.target_height:
            tmp_w = int(self.target_width / self.target_height * ori_h)
            left = (ori_w - tmp_w) // 2
            right = (ori_w + tmp_w) // 2
            image = image.crop((left, 0, right, ori_h))
        elif ori_w / ori_h < self.target_width / self.target_height:
            tmp_h = int(self.target_height / self.target_width * ori_w)
            top = ori_h - tmp_h
            bottom = ori_h
            # top = (ori_h - tmp_h) // 2
            # bottom = (ori_h + tmp_h) // 2
            image = image.crop((0, top, ori_w, bottom))
        image = image.resize((self.target_width, self.target_height), resample=Image.LANCZOS)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = preprocessor(image)
        return image

    def build_data_dict(self, image_seq, guide_seq, sample_dict): # type: ignore
        # log_cond_aug = self.log_cond_aug_dist.sample()
        # cond_aug = torch.exp(log_cond_aug)
        cond_aug = torch.tensor([0.0])
        data_dict = {
            "img_seq": torch.stack(image_seq),
            "guide_seq": torch.stack(guide_seq),
            "motion_bucket_id": torch.tensor([127]),
            "fps_id": torch.tensor([9]),
            "cond_frames_without_noise": image_seq[0],
            "cond_frames": image_seq[0] + cond_aug * torch.randn_like(image_seq[0]),
            "cond_aug": cond_aug
        }
        return data_dict

    def __getitem__(self, index):
        sample_dict = self.samples[index]

        image_seq = list()
        for i in range(self.num_frames):
            current_index = i
            img_path = self.get_image_path(sample_dict, current_index)
            image = self.preprocess_image(img_path, self.img_preprocessor)
            image_seq.append(image)
            
        guide_seq = list()
        guide_seq_path = list()

        for i in range(self.num_frames):
            current_index = i
            guide_path = self.get_guidance_path(sample_dict, current_index)
            guide = self.preprocess_image(guide_path, self.guide_preprocessor)
            guide_seq.append(guide)
            guide_seq_path.append(guide_path)

        data_dict = self.build_data_dict(image_seq, guide_seq, sample_dict)
        data_dict["guide_seq_path"] = guide_seq_path
        
        return data_dict