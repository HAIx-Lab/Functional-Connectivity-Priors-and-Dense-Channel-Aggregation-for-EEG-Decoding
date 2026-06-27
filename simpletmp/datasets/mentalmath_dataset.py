import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import os
import random
import lmdb
import pickle

def to_tensor(array):
	return torch.from_numpy(array).float()


class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            e,
            mode='train',
    ):
        super(CustomDataset, self).__init__()
        self.db = e
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]

    def __len__(self):
        return len((self.keys))

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair['sample']
        label = pair['label']
        # print(label)
        return data/100, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label)


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.dataset_dir = params.dataset_dir
        self.e = lmdb.open(self.dataset_dir, readonly=True, lock=False, readahead=True, meminit=False)

    def get_data_loader(self):
        train_set = CustomDataset(self.dataset_dir, self.e, mode='train')
        val_set = CustomDataset(self.dataset_dir, self.e, mode='val')
        test_set = CustomDataset(self.dataset_dir, self.e, mode='test')
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set)+len(val_set)+len(test_set))
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=True,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=True,
            ),
        }
        return data_loader
