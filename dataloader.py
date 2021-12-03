import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader, Dataset
import scipy.sparse
class Goodbooks(Dataset):
    def __init__(self, df, mode='training', negs=99):
        super().__init__()

        self.df = df
        self.mode = mode

        self.book_nums = max(df['item_id'])+1
        self.user_nums = max(df['user_id'])+1

        self._init_dataset()

    def _init_dataset(self):
        self.Xs = []

        self.user_book_map = {}
      #  for i in range(self.user_nums):
      #      self.user_book_map[i] = []

        data = np.ones(len(self.df))
        sp = scipy.sparse.coo_matrix((data, (self.df.user_id, self.df.item_id)))
        sp = sp.tocsr()
      
        self.user_book_map = np.split(sp.indices, sp.indptr[1:-1])

        if self.mode == 'training':
            for user in tqdm.tqdm(range(len(self.user_book_map))):

                for item in self.user_book_map[user][:-1]:
                    self.Xs.append((user, item, 1))
                    for _ in range(3):
                        while True:
                            neg_sample = random.randint(0, self.book_nums-1)
                            if neg_sample not in self.user_book_map[user]:
                                self.Xs.append((user, neg_sample, 0))
                                break

        elif self.mode == 'validation':
            for user in tqdm.tqdm(range(len(self.user_book_map))):
                if len(self.user_book_map[user]) == 0:
                    continue
                self.Xs.append((user, self.user_book_map[user][-1]))
                
    def __getitem__(self, index):

        if self.mode == 'training':
            user_id, book_id, label = self.Xs[index]
            return user_id, book_id, label
        elif self.mode == 'validation':
            user_id, book_id = self.Xs[index]

            negs = list(random.sample(
                list(set(range(self.book_nums)) - set(self.user_book_map[user_id])),
                k=99
            ))
            return user_id, book_id, torch.LongTensor(negs)
    
    def __len__(self):
        return len(self.Xs)

if __name__ == '__main__':
    df = pd.read_csv('./datasets/train_dataset.csv')
    traindataset = Goodbooks(df, 'training')
    validdataset = Goodbooks(df, 'validation')

    trainloader = DataLoader(
        traindataset, batch_size=32, shuffle=True, drop_last=False, num_workers=0)
    validloader = DataLoader(
        validdataset, batch_size=32, shuffle=True, drop_last=False, num_workers=0)
