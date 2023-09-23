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
size_pref = [(4,4),(6,6),(8,8)]



print("Total samples: ",len(digits.data))
print("Shape: ",data[0].shape)

for td in size_pref:

    
    cur_test_size = 0.2
    cur_dev_size = 0.1
    cur_train_size = 0.7
    
    
    X_train, X_test, y_train, y_test, X_dev, y_dev = split_train_dev_test(data, digits.target, 
                                                                          test_size = cur_test_size, dev_size = cur_dev_size)
                                                                          
    X_train = preprocess_data(X_train,td)
    X_dev = preprocess_data(X_dev,td)
    X_test = preprocess_data(X_test,td)
    
    best_gamma, best_crange, best_model, best_acc = tune_hparams(X_train, y_train, X_dev, y_dev, hyperparameter_combinations)
    
    train_acc = predict_and_eval(best_model, X_train, y_train)
    test_acc = predict_and_eval(best_model, X_test, y_test)
    
    print(f"image_size = {td} test_size={cur_test_size} dev_size={cur_dev_size} train_size={round(cur_train_size,1)} train_acc={train_acc:.2f} dev_acc={best_acc:.2f} test_acc={test_acc:.2f}"  )
    
 
    


#X_train, X_test, y_train, y_test, X_val, y_val = split_train_dev_test(data, digits.target, test_size = 0.20, dev_size = 0.30)

#X_train = preprocess_data(X_train)
#X_test = preprocess_data(X_test)

# Create a classifier: a support vector classifier
#clf = svm.SVC(gamma=0.001)

# Learn the digits on the train subset
#clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
#predict_and_eval(clf, X_test, y_test)

