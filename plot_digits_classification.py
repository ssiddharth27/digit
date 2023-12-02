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
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
from joblib import dump, load
import numpy as np
import warnings

# Suppress a specific type of warning
warnings.filterwarnings("ignore")



# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.tree import DecisionTreeClassifier
from utils import preprocess_data, predict_and_eval, split_train_dev_test, tune_hparams ,get_hyperpara_combo,unit_normalization

digits = datasets.load_digits()
data = digits.images
data = preprocess_data(data)
data = unit_normalization(data)

# svm
gamma_list = [0.0001, 0.0005, 0.001, 0.01, 0.1, 1]
C_list = [0.1, 1, 10, 100, 1000]
h_params={}
h_params['gamma'] = gamma_list
h_params['C'] = C_list
hyperparameter_combinations = get_hyperpara_combo(h_params)

# Decision Tree
max_depth_list = [5, 10, 15, 20, 50, 100]
h_params_tree = {}
h_params_tree['max_depth'] = max_depth_list
hyperparameter_combinations_dt = get_hyperpara_combo(h_params_tree)

test_size = 0.2
dev_size = 0.1
train_size = 0.7
results = []
runs = 5

X_train, X_test, y_train, y_test, X_dev, y_dev = split_train_dev_test(data, digits.target, 
                                                                  test_size = test_size, dev_size = dev_size)
                                                                  

train_dev_len = len(X_train) + len(X_dev)
test_len = len(X_test)
print("train + dev Size: ", train_dev_len)
print("test Size: ", test_len)
############### svm ################

for i in range(runs):
        
	best_params, best_model_path, best_acc = tune_hparams(X_train, y_train, X_dev, y_dev, hyperparameter_combinations, "svm")

	best_model = load(best_model_path)
	train_acc = predict_and_eval(best_model, X_train, y_train)
	test_acc = predict_and_eval(best_model, X_test, y_test)

	print("{}\ttest_size={:.2f} dev_size={:.2f} train_size={:.2f} train_acc={:.2f} dev_acc={:.2f} test_acc={:.2f}".format("svm", test_size, dev_size, train_size, train_acc, best_acc, test_acc))
	cur_run_results = {'model_type': "svm", 'run_index': i, 'train_acc' : train_acc, 'dev_acc': best_acc, 'test_acc': test_acc}
	results.append(cur_run_results)

############### Decision treee ################

#for i in range(runs):

#	best_params_dt ,best_model_path_dt, best_acc_dt = tune_hparams(X_train, y_train, X_dev, y_dev, hyperparameter_combinations_dt, "tree")

#	best_model_dt = load(best_model_path_dt)
#	train_acc_dt = predict_and_eval(best_model_dt, X_train, y_train)
#	test_acc_dt = predict_and_eval(best_model_dt, X_test, y_test)


#	print("{}\ttest_size={:.2f} dev_size={:.2f} train_size={:.2f} train_acc={:.2f} dev_acc={:.2f} test_acc={:.2f}".format("tree", test_size, dev_size, train_size, train_acc_dt, best_acc, test_acc_dt))
#	cur_run_results = {'model_type': "tree", 'run_index': i, 'train_acc' : train_acc_dt, 'dev_acc': best_acc_dt, 'test_acc': test_acc_dt}
#	results.append(cur_run_results)

####### Logistics #######
solvers_to_try = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

for solver in solvers_to_try:

    
    model = LogisticRegression(solver=solver, max_iter=1000)
    model.fit(X_train, y_train)
    model_filename =   "./final_exam_models/" + f"B20EE067_lr_{solver}.joblib"
    
    dump(model, model_filename)
    val_accuracy = model.score(X_dev, y_dev)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    print(f"Solver: {solver}")
    print(f"Cross-Validation Accuracy: Mean = {np.mean(scores):.4f}, Std = {np.std(scores):.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print()

    
#print("Prodcution Model Accuracy: ",test_acc)
#print("Candidate Model ACcuracy: ",test_acc_dt)

#y_pred = best_model.predict(X_test)
#cm = confusion_matrix(y_test, y_pred)
#print("Confussion Matrix Prodcution: ",cm)

#y_pred_dt = best_model_dt.predict(X_test_dt)
#cm_dt = confusion_matrix(y_test_dt, y_pred_dt)
#print("Confussion Matrix Candidate: ",cm_dt)
    
 
    
#X_train, X_test, y_train, y_test, X_val, y_val = split_train_dev_test(data, digits.target, test_size = 0.20, dev_size = 0.30)

#X_train = preprocess_data(X_train)
#X_test = preprocess_data(X_test)

# Create a classifier: a support vector classifier
#clf = svm.SVC(gamma=0.001)

# Learn the digits on the train subset
#clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
#predict_and_eval(clf, X_test, y_test)

