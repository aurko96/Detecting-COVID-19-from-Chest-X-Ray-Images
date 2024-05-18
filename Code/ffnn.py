# FFNN

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf

# Load the image data
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    'Covid19-dataset/train',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(224, 224),
    shuffle=True,
    seed=42,
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    'Covid19-dataset/test',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(224, 224),
    shuffle=False,
    seed=42,
)

# Extract the image data and labels from the data sets
train_images = np.concatenate([x for x, y in train_data], axis=0)
train_labels = np.concatenate([y for x, y in train_data], axis=0)
test_images = np.concatenate([x for x, y in test_data], axis=0)
test_labels = np.concatenate([y for x, y in test_data], axis=0)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)

# Generate predictions for the test data
test_predictions = model.predict(test_images)

# Convert the one-hot encoded test labels to integer labels
test_true_labels = np.argmax(test_labels, axis=1)

# Convert the predicted probabilities to integer predictions
test_pred_labels = np.argmax(test_predictions, axis=1)

# Calculate precision, recall, and F1 score for each class
precision = precision_score(test_true_labels, test_pred_labels, average=None, zero_division=1.0)
recall = recall_score(test_true_labels, test_pred_labels, average=None, zero_division=1.0)
f1 = f1_score(test_true_labels, test_pred_labels, average=None, zero_division=1.0)

# Calculate overall accuracy
accuracy = accuracy_score(test_true_labels, test_pred_labels)

# Print the test accuracy and classification report
class_names = label_encoder.classes_
print('\nFFNN Algorithm \n')
print('Test accuracy:', accuracy)
print('\n')
print('Classification report:')
for i in range(len(class_names)):
    print(class_names[i], 'precision:', precision[i])
    print(class_names[i], 'recall:', recall[i])
    print(class_names[i], 'F1 score:', f1[i])
print('Average precision:', np.mean(precision))
print('Average recall:', np.mean(recall))
print('Average F1 score:', np.mean(f1))