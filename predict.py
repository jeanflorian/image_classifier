#predict.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json
import argparse


def main():
   
    parser  = argparse.ArgumentParser( description = 'This is a program that predict flower name from an image')

    parser.add_argument('input', type=str, help='the path to the image to predict')
    parser.add_argument('checkpoint', type=str, help='the path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=3, help='top_k value')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='mapping of categories to real names')
    parser.add_argument('--gpu',  type=bool, default='True')

    args = parser.parse_args()
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    model = load_checkpoint(args.checkpoint)
    
    probs, classes = predict(args.input, model)

    probs = probs.cpu().numpy()[0]
    classes = classes.cpu().numpy()[0]

    class_to_idx = model.class_to_idx

    for ii in range(len(classes)):
        for key, value in class_to_idx.items():
            if key == str(classes[ii]):
                classes[ii] = value
                break

    label = []       
    for jj in range(len(classes)):
        for key, value in cat_to_name.items():
            if key == str(classes[jj]):
                label.append(value)
                break
    
    print('Most likely image class: ',label[0],'. Probability: ', probs[0])
    
    print('Top-',args.top_k,': ', label)
    
    df = pd.DataFrame(
        {'flowers': pd.Series(data=label),
         'probabilities': pd.Series(data=probs, dtype='float64')
        })
    print(df)
    
    
def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = 'cpu'
    if gpu == True:
        device ='cuda'
        
    model.eval() # inference mode
    
    # TODO: Implement the code to predict the class from an image file
    img = process_image(image_path)
    
    imgTensor = torch.from_numpy(img)
    imgTensor = imgTensor.to(device)
    
    img_reshaped = imgTensor[None, :, :, :]
    
    model = model.double()
    with torch.no_grad():
        logits = model.forward(img_reshaped)
        
    ps = F.softmax(logits, dim =1)
    
    probs, classes = ps.topk(topk)
    
    return probs, classes
      
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    im.thumbnail((256, 256))
    im = im.crop([0,0,224,224])
    np_image = np.array(im)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image= np_image/255
    np_image_norm = (np_image - mean) / std
   
    r = np_image_norm.transpose((2,0,1))
 
    return r    
    
# function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
  
    classifier = Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])

    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model    
    
    
    
    
    
    
    
    
    
    
    
    
