"""Created by:
Belal Abu-Thuraia
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from data_cleaning import load_images_p2

train_folder = os.getcwd() + '/Dataset/Train'
test_folder = os.getcwd() + '/Dataset/Test'
classes = ["Focused", "Happy", "Neutral", "Surprised"]

X_train, y_train, count_train = load_images_p2(train_folder, class_limit=400)
X_test, y_test, count_test = load_images_p2(test_folder, class_limit=100)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Selecting random 25 images from each class
random_imgs = []
for class_index in range(len(classes)):
    class_indices = np.where(y_train == class_index)[0]

    # Shuffle for randomness
    np.random.shuffle(class_indices)
    class_indices = class_indices[:25]
    random_imgs.append(class_indices)


# - Class Distribution
def distribution():
    # Count the number of images in each class
    class_counts_train = np.bincount(y_train)
    class_counts_val = np.bincount(y_val)
    class_counts_test = np.bincount(y_test)

    # Combine validation and training in this context
    combined = class_counts_val + class_counts_train
    # Define the class names
    classes = ["Focused", "Happy", "Neutral", "Surprised"]

    # Plotting the bar graph
    fig, ax = plt.subplots(figsize=(8, 5))
    index = np.arange(len(classes))
    bar_width = 0.35

    ax.bar(index, combined, bar_width, label='Train')
    ax.bar(index + bar_width, class_counts_test, bar_width, label='Test')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Number of Images')
    ax.set_title('Number of Images in Each Class')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(classes)
    ax.legend()

    plt.tight_layout()
    plt.show()


# - Sample Images
def sample_img():
    # Loop through each class
    for class_index, class_name in enumerate(classes):
        class_indices = random_imgs[class_index]

        # Create a new figure for each class
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        fig.suptitle(f'{class_name}', fontsize=24)

        # Plot the images in a 5x5 grid
        for i, idx in enumerate(class_indices):
            row = i // 5
            col = i % 5
            ax = axes[row, col]
            ax.imshow(X_train[idx], cmap='gray')
            ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


# - Pixel Intensity Distribution

def pixel_dist():
    for class_index, image_index in enumerate(random_imgs):
        image = X_train[image_index]
        plt.figure(figsize=(10, 5))
        plt.title(f'Pixel Intensity Distribution: {classes[class_index]}')

        # Calculate and plot histogram for each image
        plt.hist(image.ravel(), bins=256, color='black', alpha=0.7, label='Overall Intensity')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()


distribution()
sample_img()
pixel_dist()
