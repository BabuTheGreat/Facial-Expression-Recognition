import torch
from torch import nn, optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from Main_CNN import CNN
from data_cleaning import load_images_p3


# Function to load the model
def load_model(model_path, device):
    model = CNN(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# Function to evaluate the model using the load_images function
def evaluate_model(model, device, X, y):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X).float().to(device)
        labels = torch.tensor(y, dtype=torch.long).to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_true = labels.cpu().numpy()
        y_pred = predicted.cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    return accuracy, precision, recall, fscore


# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = './best_model.pth'
model = load_model(model_path, device)

# Evaluate for each group
attributes = ['Age', 'Gender']
groups = {
    'Age': ['Young', 'Old'],
    'Gender': ['Male', 'Female']
}

# Load images and evaluate
results = {}
for attribute in attributes:
    results[attribute] = {}
    for group in groups[attribute]:
        X, y, _ = load_images_p3(f'./Dataset_p3/{attribute}/Test/{group}')
        results[attribute][group] = evaluate_model(model, device, X, y)

# Print
for attribute, groups in results.items():
    print(f"Results for {attribute}:")
    for group, metrics in groups.items():
        print(
            f"  {group}: Accuracy: {metrics[0]}, Precision: {metrics[1]}, Recall: {metrics[2]}, F1-Score: {metrics[3]}")