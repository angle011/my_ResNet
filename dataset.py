# from data.base_dataset import BaseDataset, get_transform, get_params
# from data.image_folder import make_dataset
from PIL import Image
import os
import numpy as np
from torchvision import transforms

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
train_transform =transforms.Compose(
    [
        transforms.Resize((960,640),Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean,norm_std)
    ]
)
val_transform =transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(norm_mean,norm_std)
    ]
)

class CustomDataset():
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    datafolder-tree
    dataroot:.
            ├─A
            ├─B
            ├─label
    """

    def __init__(self, txt_path,isTrain=False):
        self.img_info = self.get_Image(txt_path)
        self.isTrain=isTrain


    def get_Image(self, txt_path):
        with open(txt_path, 'r') as f:
            img_info = f.readlines()
            img_info = list(map(lambda x: x.strip().split(), img_info))
        return img_info

    def __getitem__(self, index):
        img_path, label = self.img_info[index]
        if self.isTrain:
            A = Image.open('/data/adv/train/'+img_path).convert('RGB')
        else:
            A = Image.open('/data/adv/val/'+img_path).convert('RGB')


        # transform_params = get_params(self.opt, A_img.size, test=self.istest)
        # apply the same transform to A B L
        # transform = get_transform(self.opt, transform_params, test=self.istest)

        # A = transform(A_img)
        # B = transform(B_img)
        A = train_transform(A)

        # if self.istest:
        #     return {'A': A, 'A_paths': img_path, 'B': B, 'B_paths': B_path}

        # return {'A': A, 'A_paths': img_path,
        #         'L': label}
        return A,label
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_info)
