# -*- coding: utf-8 -*-
"""
TensorFlow_Crop_Recommendation.ipynb
"""

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import numpy as np

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def build_and_train_model():
    """
    Loads, preprocesses, builds, trains, and evaluates the crop recommendation model.
    """
    # 1. Load Data
    try:
        data = pd.read_csv('Crop_recommendation.csv')
    except FileNotFoundError:
        print("Error: 'Crop_recommendation.csv' not found.")
        print("Please make sure the file is in the same directory as the script.")
        return

    # Check for missing values (and drop them if any)
    if data.isnull().values.any():
        print("Warning: Missing values found. Dropping rows with nulls.")
        data = data.dropna()

    print(f"Data loaded successfully with {len(data)} samples.")

    # 2. Preprocess Data

    # Separate features (X) and target label (y)
    X = data.drop('label', axis=1)
    y = data['label']

    # --- Feature Scaling ---
    # Scale numerical features for the neural network
    # This is crucial for models that are sensitive to feature scales
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Label Encoding ---
    # Convert string labels (crop names) into integers
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Get the number of unique classes for the output layer
    num_classes = len(encoder.classes_)
    print(f"Found {num_classes} unique crop types.")

    # 3. Split Data
    # Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded # Ensure balanced classes in train/test splits
    )

    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    # 4. Build the Neural Network Model

    # We use a Sequential model, a simple stack of layers.
    #
    model = Sequential([
        # InputLayer is explicit about the shape (7 features)
        InputLayer(shape=(X_train.shape[1],)),

        # First hidden layer with 64 neurons and ReLU activation
        # ReLU (Rectified Linear Unit) is a common activation function
        Dense(64, activation='relu'),

        # Second hidden layer with 32 neurons
        Dense(32, activation='relu'),

        # Output layer
        # It must have 'num_classes' neurons
        # 'softmax' activation converts outputs into probabilities for each class
        Dense(num_classes, activation='softmax')
    ])

    # 5. Compile the Model
    model.compile(
        optimizer='adam',  # Adam is an efficient and popular optimizer
        loss='sparse_categorical_crossentropy', # Use this loss for integer labels
        metrics=['accuracy'] # We want to track accuracy
    )

    # Print a summary of the model's architecture
    model.summary()

    # 6. Train the Model
    print("\nStarting model training...")
    history = model.fit(
        X_train,
        y_train,
        epochs=50,          # Number of times to iterate over the entire dataset
        batch_size=32,      # Number of samples per gradient update
        validation_data=(X_test, y_test), # Data to evaluate on at the end of each epoch
        verbose=1           # Show training progress
    )
    print("Model training finished.")

    # 7. Evaluate the Model
    print("\nEvaluating model on test data...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    print("\n--- Model Evaluation ---")
    print(f"Test Loss:     {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # 8. Show a detailed classification report
    print("\n--- Classification Report ---")
    y_pred_probs = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred_probs, axis=1) # Get the class with highest prob

    # Convert integer labels back to crop names for readability
    report = classification_report(
        y_test,
        y_pred_labels,
        target_names=encoder.classes_
    )
    print(report)

if __name__ == "__main__":
    build_and_train_model()
