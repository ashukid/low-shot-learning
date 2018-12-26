import os
import cv2
import numpy as np
import time
import json

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Class separating script')
    parser.add_argument('--dataset_folder', default='LouisVuitton', help='root folder of dataset')
    parser.add_argument('--test_folder', default='LouisVuitton-test', help='root folder of test dataset')
    parser.add_argument('--test_images', default='20', help='no of test images')
    return parser.parse_args()


params=parse_args()
data=params.dataset_folder
test_folder=params.test_folder
tot_test_images=params.test_images

folders=os.listdir(data)
folders.sort()

# creating the test folder if not exist
try:
    os.makedirs(test_folder)
except FileExistsError:
    continue

# creating folders for each classes inside test folder
for f in folders:
    path=os.path.join(data,f)
    if(os.path.isdir(path)):
        try:
            os.makedirs(os.path.join(test_folder,f))
        except FileExistsError:
            continue

# copying images inside each folder            
for f in folders:
    path = os.path.join(data,f)
    images=os.listdir(path)
    
    for i in range(0,min(tot_test_images,len(images))):
        im = images[i]
        impath = os.path.join(path,im)
        newpath=os.path.join(test_folder,f+'/'+im)
        os.rename(impath,newpath)
