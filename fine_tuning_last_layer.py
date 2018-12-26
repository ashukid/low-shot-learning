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

def parse_args():
    parser = argparse.ArgumentParser(description='Class separating script')
    parser.add_argument('--trainfile', default='features/new-folder/train.hdf5', help='path for training data features')
    parser.add_argument('--num_classes',default=10378, type=int, help='number of classes')
    parser.add_argument('--batchsize',default=32, type=int, help='batch size')
    parser.add_argument('--lr',default=0.1, type=int, help='learning rate')
    parser.add_argument('--momentum',default=0.9, type=int, help='momentum')
    parser.add_argument('--wd',default=0.001, type=int, help='weight decay')
    parser.add_argument('--maxiters',default=1000, type=int, help='total iteration for fine tuning')
    
    return parser.parse_args()


params=parse_args()


with open('base_classes.json') as f:
    base_classes=json.load(f)

with open('novel_classes.json') as f:
    novel_classes=json.load(f)

trainfile=params.trainfile
num_classes=params.num_classes
batchsize=params.batchsize
lr=params.lr
momentum=params.momentum
wd=params.wd
maxiters=params.maxiters

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
    
    for it in range(maxiters):
        
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
            
            if (it%100==0):
                print('{:d}: {:f}'.format(it, loss.data[0]))
            
    return model

if __name__ == '__main__':

    with h5py.File(trainfile, 'r') as f:
        train_loader=get_loader(f)
    
    model=training_loop(train_loader)
    # save finetuned model
    torch.save(model.state_dict(),'finetuned_model.pth')