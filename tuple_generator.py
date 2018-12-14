import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Class separating script')
    parser.add_argument('--dataset_folder', default='cifar/train/root/', help='root folder of dataset')
    return parser.parse_args()


params=parse_args()
dataset_folder=params.dataset_folder
base_classes=[]
novel_classes=[]
class_folders=os.listdir(dataset_folder)
class_folders.sort()

idx=0
threshold=300
for folders in class_folders:
    path=os.path.join(dataset_folder,folders)
    if(os.path.isdir(path)):
        total_images=os.listdir(path)
        if(len(total_images)>threshold):
            base_classes.append(idx)
        else:
            novel_classes.append(idx)
    idx+=1


with open('base_classes.json','w+') as f:
    json.dump(base_classes,f)
with open('novel_classes.json','w+') as f:
    json.dump(novel_classes,f)

