import cv2, os, random
import torch
import numpy as np
from torch.utils.data import Dataset
import PIL.Image as Image
from torchvision import transforms

from util.util import RandomHorizontallyFlip, RandomVertivallyFlip


class TrainValDataset(Dataset):
    def __init__(self, dataroot=None):
        super().__init__()
        self.dir = dataroot
        self.img_path, self.gt_path = os.path.join(self.dir, 'data_low'), os.path.join(self.dir, 'data_gt')
        with open(os.path.join(self.dir, 'streetv_list.txt'), 'r') as f:
            self.path_files = f.readlines()
        self.img_view_path1 = self.path_files[0::3]
        self.img_view_path2 = self.path_files[1::3]
        self.img_view_path3 = self.path_files[2::3]
        self.gt_view_path2  = self.path_files[1::3]

        assert (len(self.img_view_path1) == len(self.img_view_path2) == len(self.img_view_path3)), 'the dataset of %s does not match.' % self.dir
        self.dataset_size = len(self.img_view_path2) # Input: different views
        # self.dataset_size = len(self.img_path_files) # Input: the same view
        # print(self.dataset_size)

        self.hflip, self.vflip = RandomHorizontallyFlip(), RandomVertivallyFlip()
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img_name = self.img_view_path2[index % self.dataset_size].strip('\n').split('/')[-1]
        # image_name = self.img_path_files[index % self.dataset_size].split('/')[-1]

        # input different views
        img_view1 = cv2.imread(os.path.join(self.img_path, self.img_view_path1[index % self.dataset_size].strip('\n')))
        img_view2 = cv2.imread(os.path.join(self.img_path, self.img_view_path2[index % self.dataset_size].strip('\n')))
        img_view3 = cv2.imread(os.path.join(self.img_path, self.img_view_path3[index % self.dataset_size].strip('\n')))
        normal_view = cv2.imread(os.path.join(self.gt_path, self.gt_view_path2[index % self.dataset_size].strip('\n')))
        
        # random crop
        h, w = img_view2.shape[0], img_view2.shape[1]
        crop_size = 96
        crop_h, crop_w = np.random.randint(0, h-crop_size), np.random.randint(0, w-crop_size)
        img_view1 = img_view1[crop_h:(crop_h + crop_size), crop_w:(crop_w + crop_size), :]
        img_view2 = img_view2[crop_h:(crop_h + crop_size), crop_w:(crop_w + crop_size), :]
        img_view3 = img_view3[crop_h:(crop_h + crop_size), crop_w:(crop_w + crop_size), :]
        normal_view = normal_view[crop_h:(crop_h + crop_size), crop_w:(crop_w + crop_size), :]

        # multi-scale training strategy
        # size = [192, 224, 256, 288, 320][np.random.randint(0, 5)]
        # I = cv2.resize(I, dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        # B = cv2.resize(B, dsize=(size, size), interpolation=cv2.INTER_LINEAR)

        # random horizonally flip
        img_view1, img_view2, img_view3, normal_view = self.hflip(img_view1, img_view2, img_view3, normal_view)

        img_view_l = [img_view1, img_view2, img_view3]
        low_views = np.stack(img_view_l, axis=0)
        low_views = np.transpose(low_views[:, :, :, ::-1], (0, 3, 1, 2)).astype(np.float32) / 255.0

        # img_view1 = np.transpose(img_view1.astype(np.float32) / 255.0, (2, 0, 1))
        # img_view2 = np.transpose(img_view2.astype(np.float32) / 255.0, (2, 0, 1))
        # img_view3 = np.transpose(img_view3.astype(np.float32) / 255.0, (2, 0, 1))
        normal_view = np.transpose(normal_view.astype(np.float32) / 255.0, (2, 0, 1))

        sample = {'low_views': low_views, 'normal_view': normal_view, 'img_name': img_name}
        return sample

    def my_collate(self, batch):
        osize = [192, 224, 256, 288, 320][np.random.randint(0, 5)]
        img, gt = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            img[i] = cv2.resize(img[i], dsize=(osize, osize), interpolation=cv2.INTER_LINEAR)
            gt[i]  = cv2.resize(gt[i] , dsize=(osize, osize), interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(np.stack(img, axis=0)).permute(0, 3, 1, 2)
        gt  = torch.from_numpy(np.stack(gt , axis=0)).permute(0, 3, 1, 2)
        return img, gt

    def __len__(self):
        return self.dataset_size


class TestDataset(Dataset):
    def __init__(self, dataroot=None):
        super().__init__()
        self.dir = dataroot
        self.img_path, self.gt_path = os.path.join(self.dir, 'data_low'), os.path.join(self.dir, 'data_gt')
        with open(os.path.join(self.dir, 'streetv_list.txt'), 'r') as f:
            self.path_files = f.readlines()
        self.img_view_path1 = self.path_files[0::3]
        self.img_view_path2 = self.path_files[1::3]
        self.img_view_path3 = self.path_files[2::3]
        self.gt_view_path2  = self.path_files[1::3]
        assert (len(self.img_view_path1) == len(self.img_view_path2) == len(self.img_view_path3)), 'the dataset of %s does not match.' % self.dir
        self.dataset_size = len(self.img_view_path2)

    def __getitem__(self, index):
        img_name = self.img_view_path2[index % self.dataset_size].strip('\n')

        # input different views
        img_view1 = cv2.imread(os.path.join(self.img_path, self.img_view_path1[index % self.dataset_size].strip('\n')))
        img_view2 = cv2.imread(os.path.join(self.img_path, self.img_view_path2[index % self.dataset_size].strip('\n')))
        img_view3 = cv2.imread(os.path.join(self.img_path, self.img_view_path3[index % self.dataset_size].strip('\n')))
        normal_view = cv2.imread(os.path.join(self.gt_path, self.gt_view_path2[index % self.dataset_size].strip('\n')))

        img_view_l = [img_view1, img_view2, img_view3]
        low_views = np.stack(img_view_l, axis=0)
        low_views = np.transpose(low_views[:, :, :, ::-1], (0, 3, 1, 2)).astype(np.float32) / 255.0

        # img_view1 = np.transpose(img_view1.astype(np.float32) / 255.0, (2, 0, 1))
        # img_view2 = np.transpose(img_view2.astype(np.float32) / 255.0, (2, 0, 1))
        # img_view3 = np.transpose(img_view3.astype(np.float32) / 255.0, (2, 0, 1))
        normal_view = np.transpose(normal_view.astype(np.float32) / 255.0, (2, 0, 1))

        # multi-scale training strategy
        # size = [192, 224, 256, 288, 320][np.random.randint(0, 5)]
        # I = cv2.resize(I, dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        # B = cv2.resize(B, dsize=(size, size), interpolation=cv2.INTER_LINEAR)

        sample = {'low_views': low_views, 'normal_view': normal_view, 'img_name': img_name}
        return sample

    def __len__(self):
        return self.dataset_size


