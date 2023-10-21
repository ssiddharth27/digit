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
from sklearn.metrics import confusion_matrix


# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.tree import DecisionTreeClassifier
from utils import preprocess_data, predict_and_eval, split_train_dev_test, tune_hparams_svc, tune_hparams_dt,get_hyperpara_combo

digits = datasets.load_digits()
data = digits.images


hyperparameters = {'gamma':[0.001,0.01,0.1,1,10,100],"c_ranges":[0.1,1,2,5,10]}
hyperparameters_dt = {'depth':[None,10,20,30],"sample_split":[2,5,10]}

hyperparameter_combinations = get_hyperpara_combo(hyperparameters)
hyperparameter_combinations_dt = get_hyperpara_combo(hyperparameters_dt)
size_pref = [(4,4),(6,6),(8,8)]



    
cur_test_size = 0.2
cur_dev_size = 0.1
cur_train_size = 0.7
td = (4,4)


X_train, X_test, y_train, y_test, X_dev, y_dev = split_train_dev_test(data, digits.target, 
                                                                  test_size = cur_test_size, dev_size = cur_dev_size)
                                                                  
X_train = preprocess_data(X_train,td)
X_dev = preprocess_data(X_dev,td)
X_test = preprocess_data(X_test,td)

best_gamma, best_crange, best_model, best_acc = tune_hparams_svc(X_train, y_train, X_dev, y_dev, hyperparameters, hyperparameter_combinations)

train_acc = predict_and_eval(best_model, X_train, y_train)
test_acc = predict_and_eval(best_model, X_test, y_test)


X_train_dt, X_test_dt, y_train_dt, y_test_dt, X_dev_dt, y_dev_dt = split_train_dev_test(data, digits.target, 
                                                                  test_size = cur_test_size, dev_size = cur_dev_size)
                                                                  
X_train_dt = preprocess_data(X_train_dt,td)
X_dev_dt = preprocess_data(X_dev_dt,td)
X_test_dt = preprocess_data(X_test_dt,td)

max_depth, min_samples_split, best_model_dt, best_acc_dt = tune_hparams_dt(X_train_dt, y_train_dt, X_dev_dt, y_dev_dt, hyperparameters_dt,  hyperparameter_combinations_dt)

train_acc_dt = predict_and_eval(best_model_dt, X_train_dt, y_train_dt)
test_acc_dt = predict_and_eval(best_model_dt, X_test_dt, y_test_dt)
    
print("Prodcution Model Accuracy: ",test_acc)
print("Candidate Model ACcuracy: ",test_acc_dt)

y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confussion Matrix Prodcution: ",cm)

y_pred_dt = best_model_dt.predict(X_test_dt)
cm_dt = confusion_matrix(y_test_dt, y_pred_dt)
print("Confussion Matrix Candidate: ",cm_dt)
    
 
    
#X_train, X_test, y_train, y_test, X_val, y_val = split_train_dev_test(data, digits.target, test_size = 0.20, dev_size = 0.30)

#X_train = preprocess_data(X_train)
#X_test = preprocess_data(X_test)

# Create a classifier: a support vector classifier
#clf = svm.SVC(gamma=0.001)

# Learn the digits on the train subset
#clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
#predict_and_eval(clf, X_test, y_test)

