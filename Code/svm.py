# SVM

import numpy as np
import os
import cv2
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

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
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
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
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        
        # Append the image and label to the test lists
        test_images.append(img)
        test_labels.append(label)

# Convert the train and test images and labels to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Flatten the image data to a 1D array
n_train_samples = train_images.shape[0]
train_images = train_images.reshape((n_train_samples, -1))

n_test_samples = test_images.shape[0]
test_images = test_images.reshape((n_test_samples, -1))

# Convert the labels to integers
le = LabelEncoder()
train_labels = le.fit_transform(train_labels)
test_labels = le.transform(test_labels)

# Define the SVM classifier and fit it to the training data
svm = SVC(kernel='linear', C=1, random_state=42)
svm.fit(train_images, train_labels)

# Use the trained classifier to make predictions on the test data
test_predictions = svm.predict(test_images)

# Calculate the evaluation metrics
accuracy = accuracy_score(test_labels, test_predictions)
precision = precision_score(test_labels, test_predictions, average='macro')
recall = recall_score(test_labels, test_predictions, average='macro')
f1 = f1_score(test_labels, test_predictions, average='macro')
report = classification_report(test_labels, test_predictions)

# Print the evaluation metrics and classification report
print('SVM Algorithm \n')
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1)
print('Classification Report:\n', report)