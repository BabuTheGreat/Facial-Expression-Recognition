"""Created by:
Belal Abu-Thuraia
"""
import os
import torch
from torchvision import transforms
from PIL import Image

from Main_CNN import CNN
from Variant1_CNN import CNN_V1
from Variant2_CNN import CNN_V2


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


# Function to preprocess input image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


# Function to classify a single image
def classify_image(model, image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities).item()

    return predicted_class, probabilities.squeeze().numpy()


classes = ["Focused", "Happy", "Neutral", "Surprised"]


# Function to classify a dataset
def classify_dataset(model, dataset_folder):
    results = []
    for idx, class_name in enumerate(classes):
        class_folder = os.path.join(dataset_folder, class_name)
        for image_name in os.listdir(class_folder):
            image_path = os.path.join(class_folder, image_name)
            predicted_class, probabilities = classify_image(model, image_path)
            results.append((image_path, class_name, predicted_class, probabilities))
    return results


if __name__ == '__main__':
    #Small menu to select which model to run
    print("Menu:")
    print("1. Variant 1")
    print("2. Variant 2")
    print("3. Main CNN")
    model_type = input("Enter which model (number from menu) you like to load: ")
    if model_type == '1':
        model = CNN_V1(len(classes))
        model = load_model(model, 'best_model_v1.pth')
    elif model_type == '2':
        model = CNN_V2(len(classes))
        model = load_model(model, 'best_model_v2.pth')
    elif model_type == '3':
        model = CNN(len(classes))
        model = load_model(model, 'best_model.pth')

    else:
        print("Invalid option. Exiting...")
        exit(1)

    # Classify an individual image
    image_path = os.getcwd() + '/Dataset/Train/Focused/focused(8).jpg'
    predicted_class, probabilities = classify_image(model, image_path)
    print(f'Predicted class: {classes[predicted_class]}')
    print(f'Class probabilities: {probabilities}')

    dataset_path = os.getcwd() + '/Dataset/Train'
    results = classify_dataset(model, dataset_path)
    for result in results:
        image_path, true_class, predicted_class, probabilities = result
        print(f'Image: {image_path}, True class: {true_class}, Predicted class: {classes[predicted_class]}')
        print(f'Class probabilities: {probabilities}')