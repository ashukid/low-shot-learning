import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import ResNetFeat
import yaml
import data
import os
import argparse
import numpy as np
import h5py
import json


with open('base_classes.json') as f:
    base_classes=json.load(f)

with open('novel_classes.json') as f:
    novel_classes=json.load(f)


trainfile='features/new-folder/train.hdf5'
testfile='features/new-folder/val.hdf5'

num_classes=10378
batch_size=32
maxiters=10
lr=0.1
momentum=0.9
wd=0.001


class SimpleHDF5Dataset:
    def __init__(self, file_handle):
        self.f = file_handle
        self.all_feats_dset = self.f['all_feats'][...]
        self.all_labels = self.f['all_labels'][...]
        self.total = self.f['count'][0]
    def __getitem__(self, i):
        return torch.Tensor(self.all_feats_dset[i,:]), int(self.all_labels[i])

    def __len__(self):
        return self.total


# simple data loader for test
def get_loader(file_handle, batch_size=1000):
    testset = SimpleHDF5Dataset(file_handle)
    data_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    return data_loader


def training_loop(data_loader):
    
    featdim = 512
    model = nn.Linear(featdim, num_classes)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, dampening=momentum, weight_decay=wd)

    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.cuda()
    
    for _ in range(maxiters):
        
        for i,(x,y) in enumerate(data_loader):

            index=0
            while True:
                if(y[index] not in novel_classes):
                    y=torch.cat([y[0:index], y[index+1:]])
                    x=torch.cat([x[0:index], x[index+1:]])
                    index-=1
                index+=1

                if(len(y)==index):
                    break

            if(len(y)==0):
                continue

            optimizer.zero_grad()

            x = Variable(x.cuda())
            y = Variable(y.cuda())

            scores = model(x)
            loss = loss_function(scores,y)
            loss.backward()
            optimizer.step()
            if (i%100==0):
                print('{:d}: {:f}'.format(i, loss.data[0]))
            
    return model


def perelement_accuracy(scores, labels):
    topk_scores, topk_labels = scores.topk(5, 1, True, True)
    label_ind = labels.cpu().numpy()
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = topk_ind[:,0] == label_ind
    top5_correct = np.sum(topk_ind == label_ind.reshape((-1,1)), axis=1)
    return top1_correct.astype(float), top5_correct.astype(float)


def testing_loop(data_loader, model):
    model = model.eval()
    top1 = None
    top5 = None
    all_labels = None
    for i, (x,y) in enumerate(data_loader):
        x = Variable(x.cuda())
        scores = model(x)
        top1_this, top5_this = perelement_accuracy(scores.data, y)
        top1 = top1_this if top1 is None else np.concatenate((top1, top1_this))
        top5 = top5_this if top5 is None else np.concatenate((top5, top5_this))
        all_labels = y.numpy() if all_labels is None else np.concatenate((all_labels, y.numpy()))

    is_novel = np.in1d(all_labels, novel_classes)
    top1_novel = np.mean(top1[is_novel])
    top5_novel = np.mean(top5[is_novel])
    return np.array([top1_novel, top5_novel])


if __name__ == '__main__':

    with h5py.File(trainfile, 'r') as f:
        train_loader=get_loader(f)
        model=training_loop(train_loader)
        
    with h5py.File(testfile, 'r') as f:
        test_loader = get_loader(f)
        accs = testing_loop(test_loader,model)
        print("\nTop 1 Novel : {}".format(accs[0]))
        print("Top 5 Novel : {}\n".format(accs[1]))
