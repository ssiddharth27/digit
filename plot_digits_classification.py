"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
from itertools import product

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from utils import preprocess_data, predict_and_eval, split_train_dev_test, tune_hparams, get_hyperpara_combo

digits = datasets.load_digits()
data = digits.images

test_dev = {'test_size': [0.1, 0.2, 0.3],'dev_size': [0.1, 0.2, 0.3]}
hyperparameters = {'gamma':[0.001,0.01,0.1,1,10,100],"c_ranges":[0.1,1,2,5,10]}


hyperparameter_combinations = get_hyperpara_combo(hyperparameters)
test_dev_combinations = get_hyperpara_combo(test_dev)


print("Total samples: ",len(digits.data))
print("Shape: ",data[0].shape)


#X_train, X_test, y_train, y_test, X_val, y_val = split_train_dev_test(data, digits.target, test_size = 0.20, dev_size = 0.30)

#X_train = preprocess_data(X_train)
#X_test = preprocess_data(X_test)

# Create a classifier: a support vector classifier
#clf = svm.SVC(gamma=0.001)

# Learn the digits on the train subset
#clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
#predict_and_eval(clf, X_test, y_test)

