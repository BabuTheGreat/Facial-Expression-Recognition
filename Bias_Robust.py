import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from Main_CNN import CNN, train
from data_cleaning import load_images_p3

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def create_loader_from_folders(folder_paths):
    combined_images = []
    combined_labels = []

    # Load data from each folder and combine
    for path in folder_paths:
        images, labels, _ = load_images_p3(path)  # load_images returns normalized data with correct dimensions
        combined_images.append(images)
        combined_labels.append(labels)

    # Concatenate all data
    all_images = np.concatenate(combined_images, axis=0)
    all_labels = np.concatenate(combined_labels, axis=0)

    # Convert to tensors and create a dataset
    return all_images, all_labels


# Function to load a model from a checkpoint
def load_model(model_path, device):
    model = CNN(4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# Function to evaluate the model
def evaluate_model(model, device, data_loader):
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    return y_true, y_pred


# Function to compute and print metrics
def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=1)
    return accuracy, precision, recall, fscore


# Define paths to biased datasets and balanced test set folders
biased_dataset_folders = {
    'Dataset_p3_lvl1': ['./Dataset_p3_lvl1/Gender/Train/Male', './Dataset_p3_lvl1/Gender/Train/Female'],
    'Dataset_p3_lvl2': ['./Dataset_p3_lvl2/Gender/Train/Male', './Dataset_p3_lvl2/Gender/Train/Female'],
    'Dataset_p3_lvl3': ['./Dataset_p3_lvl3/Gender/Train/Male', './Dataset_p3_lvl3/Gender/Train/Female'],
}
balanced_test_folders = ['./Dataset_p3/Gender/Test/Male', './Dataset_p3/Gender/Test/Female']

for name, folders in biased_dataset_folders.items():
    print(f"Training on {name}")
    X_train, y_train = create_loader_from_folders(folders)

    # Load and combine the biased training data from multiple folders
    train_images, valid_images, train_labels, valid_labels = train_test_split(X_train, y_train, test_size=0.15,
                                                                              random_state=42)
    # DataLoader for training and validation sets
    train_dataset = TensorDataset(torch.tensor(train_images), torch.tensor(train_labels))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_dataset = TensorDataset(torch.tensor(valid_images), torch.tensor(valid_labels))
    valid_loader = DataLoader(valid_dataset, batch_size=64)

    # Declare model
    model = CNN(4).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.1)
    train(model, train_loader, valid_loader, criterion, optimizer, unsqueeze=False, out=True,
          save_file=f'retrain_{name}.pth')

    # Prepare the balanced test DataLoader
    X_test, y_test = create_loader_from_folders(balanced_test_folders)
    test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long())
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Evaluate model
    print(f"Evaluating model trained on {name}")
    modell = load_model(f'retrain_{name}.pth', device=device)
    y_true, y_pred = evaluate_model(modell, device, test_loader)
    accuracy, precision, recall, fscore = compute_metrics(y_true, y_pred)
    print(f"Metrics for {name}:")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {fscore:.4f}\n")
