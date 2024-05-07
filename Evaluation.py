"""Created by:
Belal Abu-Thuraia
"""
import os
import random

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Main_CNN import CNN
from Variant1_CNN import CNN_V1
from Variant2_CNN import CNN_V2
from data_cleaning import load_images_p2
# Set random seed for Python's built-in random module
random.seed(0)

# Set random seed for NumPy
np.random.seed(0)

# Set random seed for PyTorch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
classes = ["Focused", "Happy", "Neutral", "Surprised"]


# Define function to load test dataset
def load_test_data():
    test_folder = os.getcwd() + '/Dataset/Test'
    X_test, y_test, count_test = load_images(test_folder, class_limit=100)
    return X_test, y_test, count_test


# Define function to perform inference
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test).unsqueeze(1).float()
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
    return predicted.numpy()


# Define function to generate confusion matrix
def generate_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {title}')
    plt.show()


# Load test data
X_test, y_test, count_test = load_test_data()

# Load trained models
model_files = ['best_model.pth', 'best_model_v1.pth', 'best_model_v2.pth']
models = [CNN(len(classes)), CNN_V1(len(classes)), CNN_V2(len(classes))]

# Evaluate each model
y_tests = []
y_preds = []
model_names = ["Main CNN", "Variant 1", "Variant 2"]
i = 0
for model, model_file in zip(models, model_files):
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    y_pred = evaluate_model(model, X_test, y_test)

    y_tests.append(y_test)
    y_preds.append(y_pred)

    # Confusion Matrix:
    print(f"Confusion Matrix: {model_names[i]}\n", confusion_matrix(y_test, y_pred))
    generate_confusion_matrix(y_test, y_pred, classes, model_names[i])
    i += 1

# Calculate metrics for each model
metrics = []
for y_true, y_pred in zip(y_tests, y_preds):
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=1)
    recall_micro = recall_score(y_true, y_pred, average='micro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    metrics.append([accuracy, precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro])

# Create a DataFrame to display the metrics
df = pd.DataFrame(metrics,
                  columns=['Overall Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1-score', 'Micro Precision',
                           'Micro Recall', 'Micro F1-score'])

# Add row labels for each model
df.index = ['Main Model', 'Variant 1', 'Variant 2']

# Display the DataFrame
pd.set_option('display.max_columns', None)

print(df)