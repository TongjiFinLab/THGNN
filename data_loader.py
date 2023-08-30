import os
import sys
from torch.utils import data
import pickle

class AllGraphDataSampler(data.Dataset):
    def __init__(self, base_dir, gname_list=None, data_start=None, data_middle=None, data_end=None, mode="train"):
        self.data_dir = os.path.join(base_dir)
        self.mode = mode
        self.data_start = data_start
        self.data_middle = data_middle
        self.data_end = data_end
        if gname_list is None:
            self.gnames_all = os.listdir(self.data_dir)
            self.gnames_all.sort()
        if mode == "train":
            self.gnames_all = self.gnames_all[self.data_start:self.data_middle]
        elif mode == "val":
            self.gnames_all = self.gnames_all[self.data_middle:self.data_end]
        self.data_all = self.load_state()

    def __len__(self):
        return len(self.data_all)

    def load_state(self):
        data_all = []
        length = len(self.gnames_all)
        for i in range(length):
            sys.stdout.flush()
            sys.stdout.write('{} data loading: {:.2f}%{}'.format(self.mode, i*100/length, '\r'))
            data_all.append(pickle.load(open(os.path.join(self.data_dir, self.gnames_all[i]), "rb")))
        print('{} data loaded!'.format(self.mode))
        return data_all

    def __getitem__(self, idx):
        return self.data_all[idx]