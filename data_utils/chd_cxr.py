"""Congenital Heart Disease (CHD-CXR) Dataset. Source: https://www.kaggle.com/datasets/.../chd-cxr"""
import os
import torch
import torch.utils.data as data
from PIL import Image
import logging
from .data_path import get_dataset_path


class CHD(data.Dataset):
    CLASS_NAMES = ["ASD", "PDA", "VSD", "NORMAL"]
    LABEL_MAP = {"ASD": 0, "PDA": 1, "VSD": 2, "NORMAL": 3}

    def __init__(self, transform=None, return_label=False, data_pct=1.0, split="train"):
        super().__init__()
        self.root = get_dataset_path('CHD-CXR')
        self.transform = transform
        self.return_label = return_label
        self.split = split
        self.num_classes = 4

        assert split in ('train', 'val', 'test')
        split_dir = os.path.join(self.root, split)
        if not os.path.isdir(split_dir):
            logging.warning(f'CHD split {split} not found, falling back to train')
            split_dir = os.path.join(self.root, 'train')

        self.img_list = []
        self.labels = []

        for cls_name in os.listdir(split_dir):
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            label = self.LABEL_MAP.get(cls_name.upper())
            if label is None:
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.img_list.append(os.path.join(cls_dir, fname))
                    self.labels.append(label)

        if data_pct < 1.0:
            state = torch.Generator().manual_seed(42)
            perm = torch.randperm(len(self.img_list), generator=state)
            n = int(len(self.img_list) * data_pct)
            idx = perm[:n].tolist()
            self.img_list = [self.img_list[i] for i in idx]
            self.labels = [self.labels[i] for i in idx]

        logging.info(f'CHD {split} size: {len(self.img_list)}')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = self.labels[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if not self.return_label:
            return img

        if self.split == 'train':
            return img, torch.tensor(label, dtype=torch.float)
        else:
            return img, img_path, torch.tensor(label, dtype=torch.float)


if __name__ == "__main__":
    data = CHD()
    print(data.img_list[:5])
    print(len(data))
