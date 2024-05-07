# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Data handling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
from tensorflow.keras import models, layers, optimizers, losses, metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
from pathlib import Path
import random

# Define the path to your dataset directory in Google Drive
data_dir = "skin-disease-datasaet"


# Specify the parent directory containing train and test subdirectories
train_dir = os.path.join(data_dir, "train_set")
test_dir = os.path.join(data_dir, "test_set")

# Initialize counters
total_train_images = 0
total_test_images = 0

# Count images in train set
for category in os.listdir(train_dir):
    category_dir = os.path.join(train_dir, category)
    num_images = len(os.listdir(category_dir))
    print(f"Train - {category}: {num_images} images")
    total_train_images += num_images

# Count images in test set
for category in os.listdir(test_dir):
    category_dir = os.path.join(test_dir, category)
    num_images = len(os.listdir(category_dir))
    print(f"Test - {category}: {num_images} images")
    total_test_images += num_images

# Calculate total number of images
print(f"Total train images: {total_train_images}")
print(f"Total test images: {total_test_images}")


# Building the model architecture
model = models.Sequential()
model.add(
    layers.Conv2D(
        32, (3, 3), activation="relu", padding="same", input_shape=(150, 150, 3)
    )
)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(8, activation="softmax"))

model.summary()
model.save("skin_disease_model.h5")
# Data preprocessing

# List all class subdirectories
class_subdirs = os.listdir(data_dir)

# Initialize empty lists for images and labels
train_images = []
train_labels = []

# Use ImageDataGenerator to load and preprocess images
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=20, class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(150, 150), batch_size=20, class_mode="categorical"
)

# Compile the model
# Compile the model with the same optimizer, loss function, and metrics
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model with more epochs
history = model.fit(
    train_generator,
    epochs=50,  # Increase the number of epochs to 50 (or any other desired value)
    batch_size=15,
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc}")
print(f"Test loss: {test_loss}")

# Save the model weights
# model.save_weights("skin_model_weights.h5")
#
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
#
## Train the model
# history = model.fit(
#    train_generator,
#    epochs=25,
#    batch_size=15,
# )
#
## Evaluate the model
# test_loss, test_acc = model.evaluate(test_generator)
#
## Save the model weights
model.save_weights("skin_model_.weights.h5")
