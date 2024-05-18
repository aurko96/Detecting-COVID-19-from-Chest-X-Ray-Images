# Detecting-COVID-19-from-Chest-X-Ray-Images

K-NN

1.	load image files and labels from dataset
2.	use augmentation layers to increase the size of the training dataset
3.	split the augmented dataset into augmented train and augmented validation dataset (80:20)
4.	init the KNN model with n_neighbors = 5, which means that the model will search for the 5 closest data points in the training set and base its prediction on these neighbors.
5.	Train the knn model with train augmented train dataset
6.	Predict the result using augmented train/validation/test dataset and calculate the accuracy, precision, recall, f1 for each class of data.

SVM
1.	load image files and labels from dataset
2.	use augmentation layers to increase the size of the training dataset
3.	split the augmented dataset into augmented train and augmented validation dataset (80:20)
4.	init the SVC model with kernel = ‘linear’, which means that the model will use a linear kernel function to separate the data points in the input space. With C = 1, means that the regularization strength equals to 1. With probability = True, the model can call predict_proba function.
5.	Train the SVC model with train augmented train dataset
6.	Predict the result using augmented train/validation/test dataset and calculate the accuracy, precision, recall, f1 for each class of data.

CNN
1.	load image files and labels from dataset
2.	use augmentation layers to increase the size of the training dataset
3.	split the augmented dataset into augmented train and augmented validation dataset (80:20)
4.	init the CNN model, using adam optimizer, categorical crossentropy loss function, accuracy metric, 10 epochs.
5.	Train the CNN model with train augmented train dataset
6.	Predict the result using augmented train/validation/test dataset and calculate the accuracy, precision, recall, f1 for each class of data.

FFNN
1.	load image files and labels from dataset
2.	use augmentation layers to increase the size of the training dataset
3.	split the augmented dataset into augmented train and augmented validation dataset (80:20)
4.	init the FFNN model, using adam optimizer with learning rate = 0.00001, categorical crossentropy loss function, accuracy metric, 100 epochs.
5.	Train the FFNN model with train augmented train dataset
6.	Predict the result using augmented train/validation/test dataset and calculate the accuracy, precision, recall, f1 for each class of data.

Ensemble Learning
1.	load image files and labels from dataset
2.	use augmentation layers to increase the size of the training dataset
3.	split the augmented dataset into augmented train and augmented validation dataset (80:20)
4.	wrap CNN, FFNN into KerasClassifierWrapper
5.	init four model mention above with same params with voting = ‘soft’, which predicts the class label based on the argmax of the sums of the predicted probabilities, which is recommended for an ensemble of well-calibrated classifiers.
6.	Train the Ensemble model with train augmented train dataset
7.	Predict the result using augmented train/validation/test dataset and calculate the accuracy, precision, recall, f1 for each class of data.