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
from train import Network


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
    
    #load checkpoint
    model = load_checkpoint(args.checkpoint)
    
    #predict
    probs, classes = predict(args.input, model, args.top_k, args.gpu)

    # mapping class names with cat json
    label = []       
    for jj in range(len(classes)):
        for key, value in cat_to_name.items():
            if key == str(classes[jj]):
                label.append(value)
                break
    
    #results print
    print('Most likely image class: ',label[0],'. Probability: ', probs[0])
    
    print('Top-',args.top_k,': ', label)
    
    df = pd.DataFrame(
        {'flowers': pd.Series(data=label),
         'probabilities': pd.Series(data=probs, dtype='float64')
        })
    
    print(df)
    
def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = 'cpu'
    if gpu == True:
        device ='cuda'
        
    model.eval() # inference mode
    model.to(device)
    # TODO: Implement the code to predict the class from an image file
    img = process_image(image_path)
    
    imgTensor = torch.from_numpy(img)
    imgTensor = imgTensor.to(device)
    
    img_reshaped = imgTensor[None, :, :, :]
    #img_reshaped.to('cpu')
    model = model.double()
    with torch.no_grad():
        logits = model.forward(img_reshaped)
        
    ps = F.softmax(logits, dim =1)
    
    probs, indices = ps.topk(topk)
    
    probs = probs.cpu().numpy()[0]
    indices  = indices.cpu().numpy()[0]

    idx_to_class = {v:k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[x] for x in indices]
    
    return probs, classes
      
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image)
    
    # Resize the images where shortest side is 256 pixels, keeping aspect ratio. 
    if image.width > image.height: 
        factor = image.width/image.height
        image = image.resize(size=(int(round(factor*256,0)),256))
    else:
        factor = image.height/image.width
        image = image.resize(size=(256, int(round(factor*256,0))))
    # Crop out the center 224x224 portion of the image.

    image = image.crop(box=((image.width/2)-112, (image.height/2)-112, (image.width/2)+112, (image.height/2)+112))

    # Convert to numpy array
    np_image = np.array(image)
    np_image = np_image/255
    # Normalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std 
    # Reorder dimension for PyTorch
    np_image = np.transpose(np_image, (2, 0, 1))

    #tensor_image = torch.from_numpy(image).type(torch.FloatTensor)
    
 
    return np_image    
    
# function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    
    if checkpoint['arch'].lower() == "vgg16":
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'].lower() == "resnet18":
        model = models.resnet18(pretrained=True)
    else:
        print('Sorry model architecture is either vgg16 or resnet18. Default model: vgg16')
        model = models.vgg16(pretrained=True)
        
    classifier = Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])

    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model    
    
if __name__ == '__main__': main()    
    
    
    
    
    
    
    
    
    
    
