import os
import random

import numpy as np
import torch
from skorch import NeuralNetClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from torch import optim, nn

from data_cleaning import load_images_p3

train_folder = os.getcwd() + '/Dataset/Train'
test_folder = os.getcwd() + '/Dataset/Test'
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
X_train, y_train, count_train = load_images_p3(train_folder, class_limit=400)
X_test, y_test, count_test = load_images_p3(test_folder, class_limit=100)
X = np.concatenate((X_train, X_test), axis=0)
X = X.astype(np.float32)
y = np.concatenate((y_train, y_test), axis=0)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def kfold_run(CNNN):
    net = NeuralNetClassifier(
        module=CNNN,
        module__num_classes=len(np.unique(y)),
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.Adam,
        lr=0.00001,
        batch_size=64,
        max_epochs=10,
        iterator_train__shuffle=True,
        device=device,
        verbose=0
    )

    # Set up 10-fold cross-validation
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scoring = {
        'accuracy': 'accuracy',
        'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
        'precision_micro': make_scorer(precision_score, average='micro', zero_division=0),
        'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
        'recall_micro': make_scorer(recall_score, average='micro', zero_division=0),
        'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
        'f1_micro': make_scorer(f1_score, average='micro', zero_division=0)
    }
    # Perform the cross-validation and capture results
    results = cross_validate(net, X, y, cv=kfold, scoring=scoring, return_train_score=False)

    # Output the results
    print("Test scores for each fold:")
    for metric in scoring:
        print(f"{metric}: {results['test_' + metric]}")
    print("Average scores across all folds:")
    for metric in scoring:
        print(f"Average {metric}: {np.mean(results['test_' + metric])}")

# print("-----This is main CNN-----\n")
# kfold_run(CNN)
# print("-----This is best Variant-----\n")
# kfold_run(CNN_V1)
