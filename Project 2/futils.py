import torch
from PIL import Image
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch import nn
from torchvision import transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np


def load_data(path):
    print("Loading and preprocessing data from {} ...".format(path))
    
    train_dir = path + '/train'
    valid_dir = path + '/valid'
    test_dir = path + '/test'
    
    # Define transforms for the training, validation, and testing sets
    train_transform = transforms.Compose([transforms.RandomRotation(50),
                                                  transforms.RandomResizedCrop(224),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                                       [0.229, 0.224, 0.225])])

    valid_transform = transforms.Compose([transforms.Resize(255),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(255),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transform)
    test_data = datasets.ImageFolder(test_dir, transform = test_transform)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64)
    
    print("Finished loading and preprocessing data.")
    
    return train_data, trainloader, validloader, testloader
# Building the network 
def build_network(architecture, hidden_units):
    print()
    print("Building network ... architecture: {}, hidden_units: {}".format(architecture, hidden_units))
    if architecture == 'vgg19':
      model = models.vgg19(pretrained = True)
      input_units = 25088
    elif architecture =='vgg16':
        model = models.vgg16(pretrained = True)
        input_units = 25088
    elif architecture =='vgg13':
        model = models.vgg13(pretrained = True)
        input_units = 25088
    elif architecture =='alexnet':
        model = models.alexnet(pretrained = True)
        input_units = 9216
    elif architecture =='densenet121':
        model = models.densenet121(pretrained=True)
        input_units = 1024
        
    for param in model.parameters():
        param.requires_grad = False
    
    
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_units, hidden_units)),
                                        ('dropout', nn.Dropout(0.2)),
                                        ('relu', nn.ReLU()),
                                        ('fc2', nn.Linear(4096, 1024)),
                                        ('dropout', nn.Dropout(p=0.2)),
                                        ('relu', nn.ReLU()),
                                        ('fc3', nn.Linear(1024, 256)),
                                        ('dropout', nn.Dropout(p=0.2)),
                                        ('relu', nn.ReLU()),
                                        ('out', nn.Linear(256, 102)),
                                        ('softmax', nn.LogSoftmax(dim = 1))]))

    model.classifier = classifier
    
    print("Finished building network.")
    
    return model

# Create a function to load the checkpoint
def load_checkpoint(checkpoint_path):
  device = 'cpu'
  if(torch.cuda.is_available()):
      device = 'gpu'
  checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
  model = build_network(checkpoint['architecture'], checkpoint['hidden_units'])
  model.load_state_dict(checkpoint['model_state_dict'])
  epochs = checkpoint['epochs']
  learning_rate = checkpoint['learning_rate']
  model.class_to_idx = checkpoint['class_to_idx']

  return model, epochs, learning_rate


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image_path)
    image_transform = transforms.Compose([transforms.Resize(255),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225])])

    return image_transform(image)



def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

# Predict the flower class and plot the probability graphs for different flowers
def predict(image_path, model, gpu, topk=5, category_names='cat_to_name.json'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    model.eval()
    device = "cpu"
    if(gpu=='1' and torch.cuda.is_available()):
        device = "cuda"
    model.to(device)
    print("Device used for prediction : {}".format(device))
    with torch.no_grad():
        p_img = process_image(image_path).unsqueeze(0)
        p_img = p_img.to(device)
      
        logps = model.forward(p_img)
        ps = torch.exp(logps)
        probs, top_class_index = ps.topk(topk, dim = 1)

        # Inverting the class_to_idx dictionary
        class_idx = model.class_to_idx
  
        class_to_idx_inv = {class_idx[i] : i for i in class_idx}
        top_classes_index, top_classes = [], []

        with open(category_names, 'r') as file:
          cat_names = json.load(file)
        for i in top_class_index.cpu().numpy()[0]:
          top_classes_index.append(class_to_idx_inv[i])
          top_classes.append(cat_names[class_to_idx_inv[i]])

        # Plotting the graph of the topk probabilites of flowers
        fig = plt.figure(figsize=(4, 4))
        prob = probs.cpu().numpy()[0]
        sns.barplot(y=top_classes, x=prob)
        plt.xlabel("Probability")
        plt.ylabel("Flowers")
        plt.show()
        return probs.cpu().numpy()[0], top_classes_index, top_classes