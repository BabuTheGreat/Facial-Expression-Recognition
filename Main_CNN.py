"""Created by:
Belal Abu-Thuraia
"""
import os
from data_cleaning import load_images_p3, load_images_p2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import numpy as np

'''IMPORTANT NOTE
To run this model and other variants efficiently, you need to run it on the GPU. This project utilizes CUDA.
Training will take a while running on CPU, so you please use GPU if possible. 
'''

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
classes = [ "Focused","Happy", "Neutral", "Surprised"]


# Defining CNN architecture
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(160000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Convolutional layers
        x = self.conv_layer(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Fully connected layers
        x = self.fc_layer(x)
        return x



# Training model function

def train(model, train_loader, valid_loader, criterion, optimizer,unsqueeze=True, out=True, save_file = None,  num_epoch=20, lim=5 ):
    best_loss = float('inf')
    count = 0
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            if unsqueeze:
              inputs = inputs.unsqueeze(1).float().to(device)
            else:
              inputs= inputs.float().to(device)
            labels = labels.to(device)
            output = model(inputs).to(device)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        running_loss /= len(train_loader.dataset)

        # Test with validation set
        model.eval()
        valid_loss = 0.0
        correct= 0.0
        total = 0.0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                if unsqueeze:
                  inputs = inputs.unsqueeze(1).float().to(device)
                else:
                  inputs= inputs.float().to(device)
                labels= labels.to(device)
                output = model(inputs).to(device)
                loss = criterion(output, labels)
                valid_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            valid_loss /= len(valid_loader.dataset)
            accuracy = 100. * correct / total
        '''Early stopping
        If valid loss is less than the previous best, then reset limit count to 0 and save the model. Else we increment the
        count and check if count has reached the limit.'''

        if valid_loss < best_loss:
            best_loss = valid_loss
            if save_file is not None:
                torch.save(model.state_dict(), save_file)
            count = 0
        else:
            count += 1
            if count >= lim:
                print(f'Early stopping after {epoch + 1} epochs')
                break
        if out==True:
          print(f'Epoch {epoch + 1}/{num_epoch}, Train Loss: {running_loss:.4f}, Valid Loss: {valid_loss:.4f}, Accuracy: {accuracy:.2f}%')


# Loading images from train folder
def main_train(CNNN, save_file, path = '/Dataset/Train'):
    train_folder = os.getcwd() + path
    train_images, train_labels, count_train = load_images_p2(train_folder, class_limit=400)

    # Split into training and validation
    train_images, valid_images, train_labels, valid_labels = train_test_split(train_images, train_labels, test_size=0.15,
                                                                              random_state=42)

    # DataLoader for training and validation sets
    train_dataset = TensorDataset(torch.tensor(train_images), torch.tensor(train_labels))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_dataset = TensorDataset(torch.tensor(valid_images), torch.tensor(valid_labels))
    valid_loader = DataLoader(valid_dataset, batch_size=64)

    # Declare model
    model = CNNN(len(classes)).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.1)
    # Specify the full path where you want to save the model
    save_path = os.path.join(os.getcwd(), save_file)

    train(model, train_loader, valid_loader, criterion, optimizer, save_file=save_path)

if __name__ == '__main__':
    main_train(CNN, 'best_model.pth')