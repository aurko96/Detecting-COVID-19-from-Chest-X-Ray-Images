# CNN

import numpy as np
import os
import cv2
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# Set the paths to the train and test image directories
train_path = 'Covid19-dataset/train'
test_path = 'Covid19-dataset/test'

# Set the image size for resizing
IMG_SIZE = 224

# Initialize lists for the images and labels
train_images = []
train_labels = []
test_images = []
test_labels = []

# Loop through the train subfolders and load the train images
for folder in os.listdir(train_path):
    label = folder
    
    for file in os.listdir(os.path.join(train_path,folder)):
        img_path = os.path.join(train_path,folder,file)
        
        # Read the image file and resize it
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        
        # Append the image and label to the train lists
        train_images.append(img)
        train_labels.append(label)

# Loop through the test subfolders and load the test images
for folder in os.listdir(test_path):
    label = folder
    
    for file in os.listdir(os.path.join(test_path,folder)):
        img_path = os.path.join(test_path,folder,file)
        
        # Read the image file and resize it
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        
        # Append the image and label to the test lists
        test_images.append(img)
        test_labels.append(label)

# Encode the train and test labels as integer values
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
test_labels = label_encoder.transform(test_labels)

# Convert the train and test images to numpy arrays
train_images = np.array(train_images)
test_images = np.array(test_images)

# Normalize the image data to values between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# One-hot encode the labels
num_classes = len(np.unique(train_labels))
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model with categorical crossentropy loss and Adam optimizer
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model on the training data
model.fit(train_images, train_labels, epochs=10)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels)

# Use the trained model to make predictions on the test data
test_predictions = model.predict(test_images)
test_predictions = np.argmax(test_predictions, axis=1)

# Calculate the evaluation metrics
accuracy = accuracy_score(np.argmax(test_labels, axis=1), test_predictions)
precision = precision_score(np.argmax(test_labels, axis=1), test_predictions, average='macro')
recall = recall_score(np.argmax(test_labels, axis=1), test_predictions, average='macro')
f1 = f1_score(np.argmax(test_labels, axis=1), test_predictions, average='macro')

# Generate the classification report
report = classification_report(np.argmax(test_labels, axis=1), test_predictions)

# Print the evaluation metrics and classification report
print('\nCNN Algorithm \n')
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1)
print('Classification Report:\n', report)