import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dataloader import *
from model import *

seed = 114514
np.random.seed(seed)
random.seed(seed)
BATCH_SIZE = 512

hidden_dim = 128
epochs = 10
device = torch.device('cuda:0') 

print('Reading csv.....')
df = pd.read_csv('./datasets/train_dataset.csv')
print('Constructing dataset....')
traindataset = Goodbooks(df, 'training')
validdataset = Goodbooks(df, 'validation')

trainloader = DataLoader(traindataset, batch_size=BATCH_SIZE,
                         shuffle=True, drop_last=False, num_workers=0)
validloader = DataLoader(validdataset, batch_size=BATCH_SIZE,
                         shuffle=True, drop_last=False, num_workers=0)
model = NCFModel(hidden_dim, traindataset.user_nums,
                 traindataset.book_nums).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = torch.nn.BCELoss()

loss_for_plot = []
hits_for_plot = []

for epoch in range(epochs):
    print('Start Epoch ' + str(epoch))
    losses = []
    iter_idx = 0
    for index, data in enumerate(trainloader):
        iter_idx += 1
        user, item, label = data
        user, item, label = user.to(device), item.to(
            device), label.to(device).float()
        y_ = model(user, item).squeeze()
        print(user.shape)
        print(item.shape)
        loss = crit(y_, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Iter ' + str(iter_idx) + "/" + str(len(traindataset) / BATCH_SIZE) + "loss" + str(loss))

        losses.append(loss.detach().cpu().item())
        hits = []
    for index, data in enumerate(validloader):
        user, pos, neg = data
        pos = pos.unsqueeze(1)
        all_data = torch.cat([pos, neg], dim=-1)
        output = model.predict(
            user.to(device), all_data.to(device)).detach().cpu()

        for batch in output:
            if 0 not in (-batch).argsort()[:10]:
                hits.append(0)
            else:
                hits.append(1)
    print('Epoch {} finished, average loss {}, hits@20 {}'.format(epoch,
          sum(losses)/len(losses), sum(hits)/len(hits)))
    loss_for_plot.append(sum(losses)/len(losses))
    hits_for_plot.append(sum(hits)/len(hits))
    torch.save(model.state_dict(), './model.h5')
