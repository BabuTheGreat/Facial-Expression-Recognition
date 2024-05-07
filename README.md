# Facial Expression Recognition

This Machine Learning project aims to develop a robust AI system capable of analyzing facial expressions to provide feedback on people's engagement levels. Utilizing Deep Learning and Convolutional Neural Networks (CNNs), the system categorizes expressions into four selected emotional states: Happy, Surprised, Focused, and Neutral. 

## Technologies
- __Python__: Main language used
- __PyTorch__: Building and Training the CNN models
- __Scikit_learn__: Employed for model evaluation and k-fold cross validation


## Usage

### Data Preparation
- __data_cleaning.py__: Standardizes the image data for training by resizing and normalizing.
- __data_visualization.py__: Generates visual insights into the dataset, helping understand class distributions and characteristics.

### Model Training
- __Main_CNN.py__: Script to train the main CNN model.
- __Variant1_CNN.py__ and __Variant2_CNN.py__: Train alternative CNN architectures to compare performance variations.


### Model Evaluation
- __Evaluation.py__: Evaluates the trained models using metrics such as accuracy, precision, recall, and F1-score.
- __K_FOLD.py__: Implements k-fold cross-validation to ensure the model's robustness and reliability.

### Bias Analysis and Mitigation
- __Bias_Analysis.py__: Analyzes the model for potential biases and implements strategies to mitigate them.
- __Bias_Robust.py__: Checks the robustness of the model against introduced biases.
  
### Load and Test Models
- __Load_Saved.py__: Loads saved models and performs tests on new data or validation sets.

## Datasets

  __Main Dataset__: Used to train main models. [Here](https://drive.google.com/drive/folders/1jx7xaRnGa65iYHt9nn28ZzqcLDwOVkOW?usp=drive_link)
  
  __Modified Dataset__: Added Age and Gender attributes to test biases in the model. [Here](https://drive.google.com/drive/folders/14-P-Rbwct9pFteqj_U7RJGwi4hd6qKk5?usp=drive_link)
  
  __Biased Dataset__: Train model on increasing biased datasets by reducing female images. [Level 1](https://drive.google.com/drive/folders/1bdFlrojK198wjgSCOvEUuVqiqU2oDQBB?usp=drive_link) [Level 2](https://drive.google.com/drive/folders/1t2rB__Fhpy7kifC6Ugl2ggTqLwGECkt4?usp=drive_link) [Level 3](https://drive.google.com/drive/folders/1uxcuHkefSriOXjtJp85V9TO9Q45MoMQm?usp=drive_link)
