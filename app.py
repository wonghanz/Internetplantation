# Import TensorFlow and other libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os

# Define the image size and the number of classes
IMAGE_SIZE = 224
NUM_CLASSES = 10

# Define the path to the dataset folder
DATASET_PATH = "plant_dataset"

# Load the dataset and split it into training and testing sets
def load_dataset():
  # Create empty lists to store the images and labels
  images = []
  labels = []

  # Loop through the subfolders in the dataset folder
  for folder in os.listdir(DATASET_PATH):
    # Get the label from the folder name
    label = folder

    # Loop through the images in the subfolder
    for file in os.listdir(os.path.join(DATASET_PATH, folder)):
      # Get the image path
      image_path = os.path.join(DATASET_PATH, folder, file)

      # Read the image and resize it to the desired size
      image = cv2.imread(image_path)
      image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

      # Append the image and label to the lists
      images.append(image)
      labels.append(label)

  # Convert the lists to numpy arrays
  images = np.array(images)
  labels = np.array(labels)

  # Shuffle the data randomly
  indices = np.arange(len(images))
  np.random.shuffle(indices)
  images = images[indices]
  labels = labels[indices]

  # Split the data into training and testing sets (80% for training, 20% for testing)
  split_index = int(len(images) * 0.8)
  train_images = images[:split_index]
  train_labels = labels[:split_index]
  test_images = images[split_index:]
  test_labels = labels[split_index:]

  # Return the training and testing sets
  return train_images, train_labels, test_images, test_labels

# Build the neural network model using TensorFlow Keras
def build_model():
  # Create a sequential model
  model = keras.Sequential()

  # Add a convolutional layer with 32 filters, 3x3 kernel size, ReLU activation and input shape
  model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))

  # Add a max pooling layer with 2x2 pool size
  model.add(layers.MaxPooling2D((2, 2)))

  # Add another convolutional layer with 64 filters, 3x3 kernel size and ReLU activation
  model.add(layers.Conv2D(64, (3, 3), activation="relu"))

  # Add another max pooling layer with 2x2 pool size
  model.add(layers.MaxPooling2D((2, 2)))

  # Add another convolutional layer with 128 filters, 3x3 kernel size and ReLU activation
  model.add(layers.Conv2D(128, (3, 3), activation="relu"))

  # Add another max pooling layer with 2x2 pool size
  model.add(layers.MaxPooling2D((2, 2)))

  # Add a flatten layer to convert the output to a one-dimensional vector
  model.add(layers.Flatten())

  # Add a dense layer with 256 units and ReLU activation
  model.add(layers.Dense(256, activation="relu"))

  # Add a dropout layer with 0.5 probability to reduce overfitting
  model.add(layers.Dropout(0.5))

   # Add an output layer with NUM_CLASSES units and softmax activation for multi-class classification
   model.add(layers.Dense(NUM_CLASSES, activation="softmax"))

   # Return the model
   return model

# Train and evaluate the model using TensorFlow Keras
def train_and_evaluate_model(model, train_images, train_labels, test_images, test_labels):
   # Compile the model with categorical crossentropy loss function, Adam optimizer and accuracy metric
   model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

   # Fit the model on the training set with batch size of 32 and number of epochs of 
   model.fit(train_images, train_labels, batch_size=32, epochs=10)

   # Evaluate the model on the testing set and print the results
   test_loss, test_acc = model
