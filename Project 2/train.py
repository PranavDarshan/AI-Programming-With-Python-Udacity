# imports
import argparse
import futils 
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np


# take the command line arguements
parser = argparse.ArgumentParser(description='Training a neural network on a given dataset')
parser.add_argument('data', help='Path to dataset on which the neural network should be trained on')
parser.add_argument('--gpu', choices=['0', '1'], default='1', help='True for using GPU for training')
parser.add_argument('--learning_rate', default=0.001, help='Learning rate for the classifier')
parser.add_argument('--epochs', default=5, help='Number of epochs')
parser.add_argument('--save_dir', default='', help='Path to directory where the checkpoint should be saved')
parser.add_argument('--arch',choices=['vgg19','vgg16','vgg13','alexnet','densenet121'], default='vgg16', help='Network architecture (default \'vgg16\')')
parser.add_argument('--hidden_units', default=4096, help='Number of hidden units in nueral network')

args = parser.parse_args()

epochs = int(args.epochs)
learning_rate = float(args.learning_rate)
hidden_units = int(args.hidden_units)
# loading the data from a given directory
train_data, trainloader, validloader, testloader = futils.load_data(args.data)

# building the network from pretrained models in torchvision models
model = futils.build_network(args.arch, hidden_units)
model.class_to_idx = train_data.class_to_idx

# allocating gpu if asked by user
device = torch.device("cpu")
if args.gpu == '1' and torch.cuda.is_available():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training the nueral network
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
model.to(device)


print()
print("Training network ... epochs: {}, learning_rate: {}, device used for training: {}".format(epochs, learning_rate, device))
train_losses, valid_losses = [], []
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    else:
        valid_loss = 0
        accuracy = 0

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for images, labels in validloader:
                images, labels = images.to(device), labels.to(device)
                log_ps = model(images)
                valid_loss += criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        train_losses.append(running_loss/len(trainloader))
        valid_losses.append(valid_loss/len(validloader))

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
              "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

print("Finished training network.")

# Plotting the graph of training loss and validation loss vs the epochs
print("Plotting graph of training loss and validation loss vs the epochs")
valid_l = torch.tensor(valid_losses, device = 'cpu')
print(valid_l)
e = np.arange(1,epochs+1,1)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.plot(e, train_losses, label = "Train Loss")
plt.plot(e, valid_l, label = "Validation Losses")
plt.legend()
plt.show()

# Testing the network
print("Testing network ... device used for testing: {}".format(device))
test_loss = 0
test_accuracy = 0
model.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)

        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)
        test_loss += batch_loss.item()

        # Calculate accuracy of test set
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

print("Finished evaluating the model with the test dataset")
print(f"Test loss: {test_loss/len(testloader):.3f}, "
      f"Test accuracy: {test_accuracy/len(testloader):.3f}")
running_loss = 0


# Saving the checkpoint
print("Saving model ... epochs: {}, learning_rate: {}, save_dir: {}".format(epochs, learning_rate, args.save_dir))
checkpoint = {
      'architecture': args.arch,
      'hidden_units': hidden_units,
      'epochs': epochs,
      'learning_rate': learning_rate,
      'model_state_dict': model.state_dict(),
      'class_to_idx': model.class_to_idx
}

checkpoint_path = args.save_dir + "checkpoint.pth"
torch.save(checkpoint, checkpoint_path)

print("Successfully saved checkpoint to {} directory".format(args.save_dir))
print()
print("Finished training, testing and saving the checkpoint.")
print()
print("*** Run the predict.py to predict your image ***")
