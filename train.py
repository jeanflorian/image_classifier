# train.py

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json
import argparse
from workspace_utils import active_session

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
      
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        
        super().__init__()
        
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)
    
def main():
   
    parser  = argparse.ArgumentParser( description = 'This is a program that train a  network on a dataset and save the model as a checkpoint')

    parser.add_argument('data_dir', type=str)
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='save the trained model into a file')
    parser.add_argument('--arch', type=str, default='vgg16', help='architecture used for the pre trained model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_units', type=int, nargs="*", default=[512,128], help='hidden layer units')
    parser.add_argument('--epochs', type=int, default=5, help='num of epochs')
    parser.add_argument('--gpu',  type=bool, default='True')

    args = parser.parse_args()

    data_dir = args.data_dir
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #we define train_data here to be able to get back class_to_idx after
    
     
    train_loader, valid_loader, test_loader, train_class_to_idx = load_data(train_dir, valid_dir, test_dir)
    
    # load pre-trained model and input_size of the pre trained model(vgg, resnet...) according to the model
    model, input_size = pre_trained_network(args.arch)
    
    
      # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    #set a new classifier
    model.classifier = new_classifier(input_size, args.hidden_units)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr= args.learning_rate)

    #training with validation 
    print('Beginning training phase:')
    with active_session():
        trained_model = train_model(model, criterion, optimizer, args.epochs, args.gpu, valid_loader, train_loader )
    
    #test on test data
    test_model(trained_model, test_loader,criterion, args.gpu)
    
    #saving checkpoint
    save_checkpoint(trained_model, input_size, train_class_to_idx, args.hidden_units, args.epochs, optimizer, args.save_dir)
    
    print('Operation successfully completed! Network saved!')
   
    
def save_checkpoint(model, input_size, train_class_to_idx, hidden_units, epochs, optimizer, save_dir):
   
    model.class_to_idx = train_class_to_idx

    checkpoint = {'input_size': input_size,
                  'output_size': 102,
                  'hidden_layers': [each.out_features for each in model.classifier.hidden_layers],
                  'state_dict': model.state_dict(),
                 'epochs': epochs,
                 'optimizer': optimizer.state_dict,
                 'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, save_dir)
    
    
def test_model(model, test_loader, criterion, gpu): 
    device = 'cpu'
    if gpu == True:
        device ='cuda'
        
    model.to(device)
    model.eval()
    
    print("Testing trained network on new test data")
    with torch.no_grad():
        test_loss, accuracy = validation(model, test_loader, criterion, gpu)
   
    print("Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
          "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))
    
    
    
def train_model(model, criterion, optimizer, epochs, gpu, valid_loader, train_loader ):
  
    steps = 0
    running_loss = 0
    print_every = 40
    device = 'cpu'
    
    if gpu == True:
        device = 'cuda'
    
    model.to(device)
    with active_session():
        for e in range(epochs):
            model.train()
            for inputs, labels in train_loader:
                steps += 1
                # Move input and label tensors to the GPU
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    # Eval mode for inference
                    model.eval()
                    # Gradients off for validation
                    with torch.no_grad():
                        test_loss, accuracy = validation(model, valid_loader, criterion, gpu)

                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Test Loss: {:.3f}.. ".format(test_loss/len(valid_loader)),
                          "Test Accuracy: {:.3f}".format(accuracy/len(valid_loader)))

                    running_loss = 0
                    #  training is back on
                    model.train()

    return model
 
    
def load_data(train_dir, valid_dir, test_dir):
    # data_transforms pour test et validation
    data_transforms  = transforms.Compose([transforms.Resize(255), 
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    #image_datasets = 
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    #dataloaders = 
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = 32)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 32)
    
    class_to_idx = train_data.class_to_idx
    
    return train_loader, valid_loader, test_loader, class_to_idx

def pre_trained_network(arch):
    
    if "resnet18" in arch.lower():
        print('Loading resnet18 model')
        model = models.resnet18(pretrained=True)
        input_size = model.fc.in_features
        
    elif "alexnet" in arch.lower():
        print('Loading alexnet model')
        model = models.alexnet(pretrained=True)
        input_size = alexnet.classifier[1].in_features
    else:
        print('Loading vgg16 model')
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
    
    return model, input_size
        
        

def new_classifier(input_size, hidden_units):
    # Step 2: Define a new, untrained feed-forward network as a classifier 
    classifier = Network(input_size, 102, hidden_units, drop_p=0.5)
    return classifier

def validation(model, loader, criterion, gpu):
    test_loss = 0
    accuracy = 0
    device = 'cpu'
    
    if gpu == True:
        device = 'cuda'
        
    model.to(device)
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
      
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

if __name__ == '__main__': main()






