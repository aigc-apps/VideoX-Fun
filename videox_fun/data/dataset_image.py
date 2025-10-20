import json
import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset


class CC15M(Dataset):
    def __init__(
            self,
            json_path, 
            video_folder=None,
            resolution=512,
            enable_bucket=False,
        ):
        print(f"loading annotations from {json_path} ...")
        self.dataset = json.load(open(json_path, 'r'))
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        
        self.enable_bucket = enable_bucket
        self.video_folder = video_folder

        resolution = tuple(resolution) if not isinstance(resolution, int) else (resolution, resolution)
        self.pixel_transforms = transforms.Compose([
            transforms.Resize(resolution[0]),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    
    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        video_id, name = video_dict['file_path'], video_dict['text']

        if self.video_folder is None:
            video_dir = video_id
        else:
            video_dir = os.path.join(self.video_folder, video_id)

        pixel_values = Image.open(video_dir).convert("RGB")
        return pixel_values, name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name = self.get_batch(idx)
                break
            except Exception as e:
                print(e)
                idx = random.randint(0, self.length-1)

        if not self.enable_bucket:
            pixel_values = self.pixel_transforms(pixel_values)
        else:
            pixel_values = np.array(pixel_values)

        sample = dict(pixel_values=pixel_values, text=name)
        return sample

class ImageEditDataset(Dataset):
    def __init__(
        self,
        ann_path, data_root=None,
        image_sample_size=512,
        text_drop_ratio=0.1,
        enable_bucket=False,
        enable_inpaint=False,
        return_file_name=False,
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))
    
        self.data_root = data_root
        self.dataset = dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint
        self.return_file_name = return_file_name

        # Image params
        self.image_sample_size  = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.image_transforms   = transforms.Compose([
            transforms.Resize(min(self.image_sample_size)),
            transforms.CenterCrop(self.image_sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])

    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]

        image_path, text = data_info['file_path'], data_info['text']
        if self.data_root is not None:
            image_path = os.path.join(self.data_root, image_path)
        image = Image.open(image_path).convert('RGB')

        if not self.enable_bucket:
            raise ValueError("Not enable_bucket is not supported now. ")
        else:
            image = np.expand_dims(np.array(image), 0)

        source_image_path = data_info.get('source_file_path', [])
        source_image = []
        if isinstance(source_image_path, list):
            for _source_image_path in source_image_path:
                if self.data_root is not None:
                    _source_image_path = os.path.join(self.data_root, _source_image_path)
                _source_image = Image.open(_source_image_path).convert('RGB')
                source_image.append(_source_image)
        else:
            if self.data_root is not None:
                _source_image_path = os.path.join(self.data_root, source_image_path)
            _source_image = Image.open(_source_image_path).convert('RGB')
            source_image.append(_source_image)

        if not self.enable_bucket:
            raise ValueError("Not enable_bucket is not supported now. ")
        else:
            source_image = [np.array(_source_image) for _source_image in source_image]

        if random.random() < self.text_drop_ratio:
            text = ''
        return image, source_image, text, 'image', image_path

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'image')
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'image')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                pixel_values, source_pixel_values, name, data_type, file_path = self.get_batch(idx)
                sample["pixel_values"] = pixel_values
                sample["source_pixel_values"] = source_pixel_values
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx
                if self.return_file_name:
                    sample["file_name"] = os.path.basename(file_path)
                
                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(pixel_values.size())
            mask_pixel_values = pixel_values * (1 - mask) + torch.ones_like(pixel_values) * -1 * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

        return sample

if __name__ == "__main__":
    dataset = CC15M(
        csv_path="./cc15m_add_index.json",
        resolution=512,
    )
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=0,)
    for idx, batch in enumerate(dataloader):
        print(batch["pixel_values"].shape, len(batch["text"]))