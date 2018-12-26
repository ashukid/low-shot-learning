import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import yaml
import os
import argparse
import numpy as np
from PIL import Image
import h5py
import json
import additional_transforms
import torchvision.transforms as transforms
import argparse
import ResNetFeat


def get_model(model_name, num_classes):
    model_dict = dict(ResNet10 = ResNetFeat.ResNet10,
                ResNet18 = ResNetFeat.ResNet18,
                ResNet34 = ResNetFeat.ResNet34,
                ResNet50 = ResNetFeat.ResNet50,
                ResNet101 = ResNetFeat.ResNet101)
    return model_dict[model_name](num_classes, False)


def parse_transform(transform_type, transform_params):
    if transform_type=='ImageJitter':
        method = additional_transforms.ImageJitter(transform_params['jitter_params'])
        return method
    method = getattr(transforms, transform_type)
    if transform_type=='RandomSizedCrop' or transform_type=='CenterCrop':
        return method(transform_params['image_size'])
    elif transform_type=='Scale':
        return method(transform_params['scale'])
    elif transform_type=='Normalize':
        return method(mean=transform_params['mean'], std=transform_params['std'])
    else:
        return method()
    

def image_loader(image_name,transform):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = transform(image).float()
    image = Variable(image)
    image = image.unsqueeze(0)
    return image.cuda()  #assumes that you're using GPU

def get_features(model,image):
    
    scores, feats = model(image)
    return feats.data.cpu().numpy()

def testing_loop(model,x):
    x = torch.tensor(x)
    x = Variable(x.cuda())
    scores = model(x)
    print('\nPrediction Class : {}\n'.format(np.argmax(scores.data).cpu().numpy()))


def parse_args():
    parser = argparse.ArgumentParser(description='Class separating script')
    parser.add_argument('--model',default='ResNet10',help='base model')
    parser.add_argument('--config', required=True, help='image transformation parameters for testing')
    parser.add_argument('--image_path', required=True, help='testing image path')
    parser.add_argument('--modelfile', required=True, help='base trained model')
    parser.add_argument('--num_classes', default=10378,type=int, help='number of classes')
    
    
    
    return parser.parse_args()

params=parse_args()
with open(params.config,'r') as f:
    test_params = yaml.load(f)

transform_params=test_params['transform_params']
transform_list = [parse_transform(x, transform_params) for x in transform_params['transform_list']]
transform = transforms.Compose(transform_list)
image = image_loader(params.image_path,transform)

# loading base class model and getting features
model = get_model(params.model, params.num_classes)
model = model.cuda()
model = torch.nn.DataParallel(model)
tmp = torch.load(params.modelfile)
if ('module.classifier.bias' not in model.state_dict().keys()) and ('module.classifier.bias' in tmp['state'].keys()):
        tmp['state'].pop('module.classifier.bias')
    
model.load_state_dict(tmp['state'])
model.eval()
image_feature=get_features(model,image)

# loading finetuned model and getting prediction
finetune_model= nn.Linear(image_feature.shape[1], params.num_classes)
finetune_model = finetune_model.cuda()
tmp=torch.load('finetuned_model.pth')
finetune_model.load_state_dict(tmp)
finetune_model.eval()

testing_loop(finetune_model,image_feature)