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

cfg='train_save_data.yaml'
val_cfg='val_save_data.yaml'
modelfile='checkpoints/ResNet10_sgm/19.tar'
model='ResNet10'
num_classes=10
batch_size=16
maxiters=1000
lr=0.1
momentum=0.9
wd=0.001


def get_model(model_name, num_classes):
    model_dict = dict(ResNet10 = ResNetFeat.ResNet10,
                ResNet18 = ResNetFeat.ResNet18,
                ResNet34 = ResNetFeat.ResNet34,
                ResNet50 = ResNetFeat.ResNet50,
                ResNet101 = ResNetFeat.ResNet101)
    return model_dict[model_name](num_classes, False)


def get_features(model,data_loader):
    
    feature_set=[]
    label_set=[]
    for i, (x,y) in enumerate(data_loader):
        
        
        # ignoriang the data that belong to base class
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
            
        
        if i%10 == 0:
            print('{:d}/{:d}'.format(i, len(data_loader)))
        x = x.cuda()
        x_var = Variable(x)
        
        scores, feats = model(x_var)
        feature_set.extend(feats.data.cpu().numpy())
        label_set.extend(y.cpu().numpy())
        
    return (np.array(feature_set),np.array(label_set))




def training_loop(features,labels, num_classes, lr, momentum, wd, batchsize=1000, maxiters=1000):
    featdim = features.shape[1]
    model = nn.Linear(featdim, num_classes)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, dampening=momentum, weight_decay=wd)

    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.cuda()
    
    for i in range(maxiters):
        idx=i%len(labels)
        (x,y) = torch.tensor(np.array([features[idx]])),torch.tensor(np.array([labels[idx]]))
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


def testing_loop(one_shot_model,val_features,val_labels):
    one_shot_model=one_shot_model.eval()
    
    total=0
    for i in range(len(val_features)):
        idx=i%len(val_labels)
        (x,y) = torch.tensor(np.array([val_features[idx]])),torch.tensor(np.array([val_labels[idx]]))
        
        x = Variable(x.cuda())
        scores = one_shot_model(x)
        x=(np.argmax(scores.data)==y[0]).data.numpy()
        total = total + x
        
    acc=total/len(val_features)
    print('\n---> mean accuracy : {:.2f}%'.format(acc*100))

    
if __name__ == '__main__':
    with open(cfg,'r') as f:
        data_params = yaml.load(f)

    data_loader = data.get_data_loader(data_params)
    
    with open(val_cfg,'r') as f:
        val_params = yaml.load(f)
    val_loader = data.get_data_loader(val_params)

    model = get_model(model, num_classes)
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    tmp = torch.load(modelfile)
    if ('module.classifier.bias' not in model.state_dict().keys()) and ('module.classifier.bias' in tmp['state'].keys()):
        tmp['state'].pop('module.classifier.bias')
    
    model.load_state_dict(tmp['state'])
    model.eval()
    
    # one shot training data with shuffle
    feature_set,label_set=get_features(model,data_loader)
    idx=np.arange(len(feature_set))
    np.random.shuffle(idx)
    feature_set=feature_set[idx]
    label_set=label_set[idx]
    
    # one shot validation data with shuffle
    val_feature_set,val_label_set = get_features(model,val_loader)
    idx=np.arange(len(val_feature_set))
    np.random.shuffle(idx)
    val_feature_set=val_feature_set[idx]
    val_label_set=val_label_set[idx]
    
    one_shot_model = training_loop(feature_set,label_set, num_classes, lr, momentum, wd, batch_size, maxiters)
    
    testing_loop(one_shot_model,val_feature_set,val_label_set)

