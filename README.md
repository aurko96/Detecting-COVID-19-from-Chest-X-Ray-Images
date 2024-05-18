# Detecting-COVID-19-from-Chest-X-Ray-Images

**Abstract:** Automated detection of COVID-19 from chest X-rays is a promising approach for managing the COVID-19
pandemic. Machine learning techniques, particularly image classification algorithms, have shown potential for accurate diagnosis
of COVID-19 from chest X-rays. In this study, the performance of several machine learning models were evaluated, including
convolutional neural networks (CNNs), feed forward neural networks (FFNNs), support vector machines (SVMs), and k-nearest neighbor (K-NN) algorithms, 
in classifying chest X-rays into three categories: COVID-19, viral pneumonia, and normal. 
The concept of ensemble learning was explored, which involves combining multiple models to improve accuracy and reliability. The dataset consisted of chest X-rays of COVID-19, viral pneumonia, and normal lungs. 
The performance of each model were evaluated using metrics such as precision,
recall, and F1 score. The results showed that the CNN model
outperforms other models in classifying chest X-rays. Ensembles
of models also shows better result.

**An image of the dataset sample is shown below:**

![image](https://github.com/aurko96/Detecting-COVID-19-from-Chest-X-Ray-Images/assets/17502087/9a727edc-055a-4b9f-b938-351f6c0a64e8)

**The results obtained are shown below:**

![image](https://github.com/aurko96/Detecting-COVID-19-from-Chest-X-Ray-Images/assets/17502087/bc812321-5dd5-4c47-b023-35aa4ee2a368)

![image](https://github.com/aurko96/Detecting-COVID-19-from-Chest-X-Ray-Images/assets/17502087/00dfcf38-90ff-4859-a6bc-220ed774b608)


**The methodology of each models are pointed down below:**

**K-NN**

1.	Load image files and labels from dataset
2.	Use augmentation layers to increase the size of the training dataset
3.	Split the augmented dataset into augmented train and augmented validation dataset (80:20)
4.	Init the KNN model with n_neighbors = 5, which means that the model will search for the 5 closest data points in the training set and base its prediction on these neighbors.
5.	Train the knn model with train augmented train dataset
6.	Predict the result using augmented train/validation/test dataset and calculate the accuracy, precision, recall, f1 for each class of data.

**SVM**

1.	Load image files and labels from dataset
2.	Use augmentation layers to increase the size of the training dataset
3.	Split the augmented dataset into augmented train and augmented validation dataset (80:20)
4.	Init the SVC model with kernel = ‘linear’, which means that the model will use a linear kernel function to separate the data points in the input space. With C = 1, means that the regularization strength equals to 1. With probability = True, the model can call predict_proba function.
5.	Train the SVC model with train augmented train dataset
6.	Predict the result using augmented train/validation/test dataset and calculate the accuracy, precision, recall, f1 for each class of data.

**CNN**

1.	Load image files and labels from dataset
2.	Use augmentation layers to increase the size of the training dataset
3.	Split the augmented dataset into augmented train and augmented validation dataset (80:20)
4.	Init the CNN model, using adam optimizer, categorical crossentropy loss function, accuracy metric, 10 epochs.
5.	Train the CNN model with train augmented train dataset
6.	Predict the result using augmented train/validation/test dataset and calculate the accuracy, precision, recall, f1 for each class of data.

**FFNN**

1.	Load image files and labels from dataset
2.	Use augmentation layers to increase the size of the training dataset
3.	Split the augmented dataset into augmented train and augmented validation dataset (80:20)
4.	Init the FFNN model, using adam optimizer with learning rate = 0.00001, categorical crossentropy loss function, accuracy metric, 100 epochs.
5.	Train the FFNN model with train augmented train dataset
6.	Predict the result using augmented train/validation/test dataset and calculate the accuracy, precision, recall, f1 for each class of data.

**Ensemble Learning**

1.	Load image files and labels from dataset
2.	Use augmentation layers to increase the size of the training dataset
3.	Split the augmented dataset into augmented train and augmented validation dataset (80:20)
4.	Wrap CNN, FFNN into KerasClassifierWrapper
5.	Init four model mention above with same params with voting = ‘soft’, which predicts the class label based on the argmax of the sums of the predicted probabilities, which is recommended for an ensemble of well-calibrated classifiers.
6.	Train the Ensemble model with train augmented train dataset
7.	Predict the result using augmented train/validation/test dataset and calculate the accuracy, precision, recall, f1 for each class of data.
