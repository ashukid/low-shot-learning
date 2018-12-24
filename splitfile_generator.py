import h5py    
import numpy as np 
import os
import json

f1 = h5py.File('features/ResNet10_sgm/train.hdf5','r+') 
data_dict={}
labels=list(f1['all_labels'])

for idx in range(len(labels)):
    label=labels[idx]
    c=data_dict.setdefault(label,[])
    data_dict[label].append(idx)

nshot=20
unique_label=np.unique(labels)
print(len(unique_label))
data_index=[]
for i in range(len(unique_label)):
    data_index.append(data_dict[i][:nshot])

with open('experiment_cfgs/splitfile_1.json','w+') as f:
    json.dump(data_index,f)
