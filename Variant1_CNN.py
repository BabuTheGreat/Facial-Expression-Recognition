"""Created by:
Belal Abu-Thuraia
"""
import torch.nn as nn

from Main_CNN import train, main_train

classes = ["Focused", "Happy", "Neutral", "Surprised"]


# Defining CNN architecture
class CNN_V1(nn.Module):
    def __init__(self, num_classes):
        super(CNN_V1, self).__init__()
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
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # Added conv layer
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),  # Added conv layer
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(160000, 512),  # Adjusted input size for fully connected layers
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


if __name__ == "__main__":
    main_train(CNN_V1, save_file='best_model_v1.pth')
