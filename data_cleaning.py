import os
import cv2
import numpy as np
import time
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Class separating script')
    parser.add_argument('--dataset_folder', default='LouisVuitton', help='root folder of dataset')
    return parser.parse_args()


params=parse_args()
data=params.dataset_folder
folders=os.listdir(data)

count=0
t1=time.time()
for f in folders:
    
    path = os.path.join(data,f)
    if(os.path.isdir(path)):
        
        images=os.listdir(path)
        
        for im in images:
            count +=1 
            impath = os.path.join(path,im)
            x=cv2.imread(impath)
            if(type(x) is not np.ndarray):
                print(impath)
                os.remove(impath)
        
            if(count%1000==0):
                t2=time.time()
                print("Total time taken to read : {} images = {}".format(count,t2-t1))
                t1=time.time()
                count=0




