"""Created by:
Belal Abu-Thuraia
"""

import os
import random

import numpy as np
from skimage.exposure import exposure
from skimage.io import imread
from skimage.transform import resize, rotate
from skimage.color import rgb2gray
from skimage.util import img_as_float, crop

# Defining Paths for datasets
train_folder = os.getcwd() + '/Dataset/Train'
test_folder = os.getcwd() + '/Dataset/Test'

# List classes to iterate
classes = ["Focused", "Happy", "Neutral", "Surprised"]


def load_images_p2(folder_path, class_limit=None):
    images = []
    labels = []
    class_counts = {}
    for idx, class_name in enumerate(classes):
        class_folder = os.path.join(folder_path, class_name)
        class_images = os.listdir(class_folder)

        # Check if class limit is specified and reduce the number of images accordingly
        if class_limit is not None and len(class_images) > class_limit:
            class_images = random.sample(class_images, class_limit)

        class_counts[class_name] = len(class_images)  # Store the number of images for this class

        for image_name in class_images:
            image_path = os.path.join(class_folder, image_name)
            image = imread(image_path)

            # Standardize Images to common dimensions
            image = resize(image, (100, 100))  # Resize images to a common size
            if image.ndim == 3 and image.shape[2] == 3:  # RGB image
                image = rgb2gray(image)

            # Histogram equalization to enhance image contrast
            image = exposure.equalize_hist(image)
            # Convert to floating point
            image = img_as_float(image)

            # Data Augmentation to increase robustness to dataset: Crop images and rotate them
            cropped_image = crop(image, ((5, 5), (5, 5)), copy=False)
            rotated_cropped_image = rotate(cropped_image, angle=np.random.uniform(-10, 10), mode='edge')

            images.extend([image, rotated_cropped_image])

            # Assign label based on class index
            labels.extend([idx, idx])

    images = [resize(img, (100, 100)) for img in images]
    images = np.array(images)
    labels = np.array(labels)

    # Normalize pixel values
    images = images / 255.0
    return images, labels, class_counts
def load_images_p3(folder_path, class_limit=None):
    images = []
    labels = []
    class_counts = {}
    for idx, class_name in enumerate(classes):
        class_folder = os.path.join(folder_path, class_name)
        class_images = os.listdir(class_folder)

        # Check if class limit is specified and reduce the number of images accordingly
        if class_limit is not None and len(class_images) > class_limit:
            class_images = random.sample(class_images, class_limit)

        class_counts[class_name] = len(class_images)  # Store the number of images for this class

        for image_name in class_images:
            image_path = os.path.join(class_folder, image_name)
            if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
              continue

            image = imread(image_path)

            # Standardize Images to common dimensions
            image = resize(image, (100, 100))  # Resize images to a common size
            if image.ndim == 3 and image.shape[2] == 3:  # RGB image
                image = rgb2gray(image)

            # Histogram equalization to enhance image contrast
            image = exposure.equalize_hist(image)
            # Convert to floating point
            image = img_as_float(image)

            # Data Augmentation to increase robustness to dataset: Crop images and rotate them
            cropped_image = crop(image, ((5, 5), (5, 5)), copy=False)
            rotated_cropped_image = rotate(cropped_image, angle=np.random.uniform(-10, 10), mode='edge')

            images.extend([image, rotated_cropped_image])

            # Assign label based on class index
            labels.extend([idx, idx])

    images = [resize(img, (100, 100)) for img in images]
    images = np.array([img if img.ndim == 3 else img[:, :, np.newaxis] for img in images])
    images = images.transpose((0, 3, 1, 2))  # Correct the dimension order

    labels = np.array(labels)

    # Normalize pixel values
    images = images / 255.0
    return images, labels, class_counts

if __name__ == '__main__':
    classes = ["Focused", "Happy", "Neutral", "Surprised"]
    X_train, y_train, count_train = load_images_p3(train_folder, class_limit=420)
    X_test, y_test, count_test = load_images_p3(test_folder, class_limit=105)
    X_combined = np.concatenate((X_train, X_test), axis=0)
    y_combined = np.concatenate((y_train, y_test), axis=0)
    for i in classes:
        print(f"Number of images in {i} (training): {count_train[i]}")
    for i in classes:
        print(f"Number of images in {i} (test): {count_test[i]}")
    print("X_combined shape:", X_combined.shape)
    print("X_train shape: ", X_train.shape)