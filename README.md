# Crop Recommendation System using Neural Networks

This notebook implements a crop recommendation system using a feed-forward neural network. It trains a model to suggest the most suitable crop for cultivation based on various environmental parameters.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Model Architecture](#model-architecture)
- [Results](#results)

## Project Overview

The goal of this project is to build and train a deep learning model that can recommend the optimal crop to plant given specific environmental conditions such as Nitrogen, Phosphorus, Potassium levels, temperature, humidity, pH value, and rainfall.

## Dataset

The model is trained on the `Crop_recommendation.csv` dataset, which contains environmental parameters and the corresponding recommended crop label.

## Features

The notebook performs the following key steps:

1.  **Data Loading**: Loads the `Crop_recommendation.csv` dataset into a Pandas DataFrame.
2.  **Data Preprocessing**: 
    - Handles missing values (drops rows with nulls).
    - Scales numerical features using `StandardScaler`.
    - Encodes categorical crop labels into numerical format using `LabelEncoder`.
3.  **Data Splitting**: Divides the dataset into training and testing sets (80% train, 20% test) with stratification to ensure balanced class representation.
4.  **Model Building**: Constructs a sequential neural network using TensorFlow/Keras with:
    - An `InputLayer` matching the number of features.
    - Two `Dense` hidden layers with ReLU activation.
    - An output `Dense` layer with `softmax` activation for multi-class classification.
5.  **Model Compilation**: Configures the model with the 'adam' optimizer, 'sparse_categorical_crossentropy' loss function, and 'accuracy' metric.
6.  **Model Training**: Trains the neural network on the preprocessed training data for 50 epochs.
7.  **Model Evaluation**: Evaluates the trained model's performance on the test set, reporting test loss, test accuracy, and a detailed classification report.

## Technologies Used

-   **Python**
-   **Pandas**: For data loading and manipulation.
-   **NumPy**: For numerical operations.
-   **TensorFlow / Keras**: For building and training the neural network.
-   **Scikit-learn**: For data preprocessing (scaling, label encoding) and model evaluation (classification report).

## How to Run

1.  **Download the Notebook**: Save this notebook (`.ipynb` file).
2.  **Upload to Google Colab**: Open Google Colab and upload the notebook.
3.  **Upload Dataset**: Ensure the `Crop_recommendation.csv` file is present in the same directory as the notebook or upload it to your Colab environment.
4.  **Run All Cells**: Execute all cells in the notebook (`Runtime` -> `Run all`) to load data, preprocess, train, and evaluate the model.

## Model Architecture

The model is a simple feed-forward neural network:

-   **Input Layer**: Matches the number of features (e.g., 7 environmental parameters).
-   **Hidden Layer 1**: 64 neurons, ReLU activation.
-   **Hidden Layer 2**: 32 neurons, ReLU activation.
-   **Output Layer**: `num_classes` neurons (where `num_classes` is the number of unique crop types), softmax activation.

## Results

The notebook will output:
-   Data loading and preprocessing summaries.
-   Model summary (architecture details).
-   Training progress (loss and accuracy per epoch).
-   Final test loss and accuracy.
-   A detailed classification report showing precision, recall, f1-score, and support for each crop type.
