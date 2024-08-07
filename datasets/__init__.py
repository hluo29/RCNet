import os, random
import torch

from datasets.Streetv_dataset import TrainValDataset, TestDataset


def create_datasets(opt):
    data_loader = CustomDatasetDataLoader(opt.dataroot, opt.batch_size, opt.mode)
    dataset     = data_loader.load_data()
    return dataset

class CustomDatasetDataLoader():
    def __init__(self, dataroot, batch_size, train_mode=True):
        dataroot = os.path.join(dataroot, 'data_train' if train_mode else 'data_test')
        self.dataset = {
            True : TrainValDataset,
            False: TestDataset}[train_mode](dataroot=dataroot)
        print('Dataset for %s was created' % ('training' if train_mode else 'test'))
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size  = batch_size if train_mode else 1,
            shuffle     = True,
            num_workers = 16)

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data

