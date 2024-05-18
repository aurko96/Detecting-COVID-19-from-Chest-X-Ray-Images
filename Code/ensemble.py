# Ensemble Learning

import os
import cv2
import numpy as np
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from keras.wrappers.scikit_learn import KerasClassifier

# Set up the paths to the train and test directories
train_dir = 'Covid19-dataset/train'
test_dir = 'Covid19-dataset/test'

# Set up the classes and the image size
classes = ['Covid', 'Viral Pneumonia', 'Normal']
img_size = 128

# Load the training data
train_data = []
train_labels = []
for cls in classes:
    path = os.path.join(train_dir, cls)
    class_num = classes.index(cls)
    for img in os.listdir(path):
        img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        img_arr = cv2.resize(img_arr, (img_size, img_size))
        img_arr = img_arr.flatten()  # Flatten the input image
        train_data.append(img_arr)
        train_labels.append(class_num)

# Load the testing data
test_data = []
test_labels = []
for cls in classes:
    path = os.path.join(test_dir, cls)
    class_num = classes.index(cls)
    for img in os.listdir(path):
        img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        img_arr = cv2.resize(img_arr, (img_size, img_size))
        img_arr = img_arr.flatten()  # Flatten the input image
        test_data.append(img_arr)
        test_labels.append(class_num)

# Convert the data to numpy arrays
train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)

# Initialize the classifiers
knn = KNeighborsClassifier(n_neighbors=3)
svm = SVC(kernel='linear', probability=True)
ffn = MLPClassifier(hidden_layer_sizes=(128,), activation='relu', solver='adam', max_iter=1000)

# Define CNN model
cnn_model = Sequential()
cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dense(num_classes, activation='softmax'))
cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#estimator = KerasClassifier(build_fn=cnn_model, epochs=10, batch_size=32)

# Create the ensemble classifier

# TODO: Need to figure out how to implement CNN in the ensemble classifier as its showing error if we add the CNN model in the voting classifier
ensemble = VotingClassifier(estimators=[('knn', knn), ('svm', svm), ('ffn', ffn)], voting='soft')

# Train the ensemble classifier
ensemble.fit(train_data, train_labels)

# Test the ensemble classifier
predictions = ensemble.predict(test_data)
print('Ensemble Learning \n')
print('Classification report:')
print(classification_report(test_labels, predictions, target_names=classes))