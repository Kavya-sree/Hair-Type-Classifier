# Hair-Type-Classifier
A Deep learning model built using the FastAI library that classifies images of hair into five types: Straight, Wavy, Curly, Kinky and Dreadlocks. The project includes a Gradio-based web interface for easy interaction with the model.


You can find web interface here: https://kavyasree-hair-type.hf.space

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [About classifier](#about-classifier)
- [Model](#model)
- [Installation](#installation)
- [Usage](#usage)


## Project Overview
This project aims to provide an easy-to-use interface for classifying hair types based on input images. It uses a pre-trained ResNet18 model fine-tuned on a custom dataset of hair images. The model can classify images into one of four hair types with high accuracy.

## Features
- FastAI based model using ResNet18 architecture.
- Gradio for building a simple web interface to interact with the model.
- Classifies images into the following hair types: Curly, Straight, Wavy, Kinky, and Dreadlocks

## Dataset
To collect image data you can follow along this article: https://machinelearningbrain.website/image-data-collection-for-deep-learning-projects/. I have uploaded the dataset on [Kaggle](https://www.kaggle.com/datasets/kavyasreeb/hair-type-dataset/data).

## About classifier
The four main hair types are Straight, Wavy, Curly and Kinky. But in the classifier also identifies dreadlocks. Dreadlocks are not a hair type, but a hair style. Thus our Hair Type Classifier identifies 5 hair type
1. Straight
2. Wavy
3. Curly
4. Kinky
5. Dreadlocks

Figuring out your hairtype is the first step in the hair care routine.

![Screenshot (3)](https://github.com/Kavya-sree/Hair-Type-Classifier/assets/27502670/94ca1a9e-749b-4748-828c-d2dce5b1ef36)

After uploading the image, you get the predicted hair type

![Screenshot (4)](https://github.com/Kavya-sree/Hair-Type-Classifier/assets/27502670/dc9aba73-2d82-42c3-a30b-843c8909570f)


## Model

The model uses a ResNet18 architecture, fine-tuned for hair type classification. Additionally, Albumentations is used for custom augmentations during training.
**Performance**: The model achieves an accuracy of 88% on the validation set.

## Installation

1. Clone the repository:

 ```bash
    git clone https://github.com/Kavya-sree/Hair-Type-Classifier.git
```
2. Create a virtual environment and activate it:

```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. Install dependencies:

```bash
    pip install -r requirements.txt
```


## Usage

1. **Training the Model**: The model was trained using a Jupyter Notebook. To train the model from scratch:

- Open the notebook Hair_type_classifier.ipynb located in the repository.
- Run through the cells to load data, train the model, and save the trained model as a .pkl file.
- The notebook will save the trained model to the models/ directory for later use.

2. **Gradio App**: Launch the Gradio app to interact with the model:

```bash
    gradio app.py
```




