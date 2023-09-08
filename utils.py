from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

def preprocess_data(data):
   n_samples = len(data)
   data = data.reshape((n_samples,-1))
   return data

def split_train_dev_test(X, y, test_size, dev_size):

   # calculating the size of training data
   train_size = 1.0 - test_size - dev_size

   X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size= test_size,
   shuffle = False, random_state = 42)
   
   new_dev_size = dev_size / (dev_size + train_size)
   
   X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size= new_dev_size,
   shuffle = False, random_state = 42)
   
   return X_train, X_test, y_train, y_test, X_val, y_val
   
def predict_and_eval(model, X_test, y_test):

   # prediction
   predicted = model.predict(X_test)
   
   accuracy = metrics.accuracy_score(y_test, predicted)
   
   return accuracy
   
def tune_hparams(X_train, Y_train, X_dev, y_dev, list_of_all_param_combinations):

   best_acc = -1  
   hyperparameters = {'gamma':[0.001,0.01,0.1,1,10,100],"c_ranges":[0.1,1,2,5,10]} 
   
   for params in list_of_all_param_combinations:

        hyperparameter_settings = dict(zip(hyperparameters.keys(), params))

        cur_gamma = hyperparameter_settings['gamma']
        cur_crange = hyperparameter_settings['c_ranges']
	
        cur_model = svm.SVC(gamma=cur_gamma,C = cur_crange)
        cur_model.fit(X_train, Y_train)

        cur_dev_acc = predict_and_eval(cur_model, X_dev, y_dev)
        if cur_dev_acc > best_acc:
            best_acc = cur_dev_acc
            best_gamma = cur_gamma
            best_crange = cur_crange
            best_model = cur_model
	    
   return best_gamma, best_crange, best_model, best_acc
	    
	 

	#print(f"test_size={cur_test_size} dev_size={cur_dev_size} train_size={cur_train_size} train_acc={train_acc:.2f}     dev_acc={dev_acc:.2f} test_acc={test_acc:.2f}"  )
   
   # metrics
   #print(
    #f"Classification report for classifier {model}:\n"
   # f"{metrics.classification_report(y_test, predicted)}\n"
    #) 
    
   #disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
   #disp.figure_.suptitle("Confusion Matrix")
   #print(f"Confusion matrix:\n{disp.confusion_matrix}")

   # The ground truth and predicted lists
   #y_true = []
   #y_pred = []
   #cm = disp.confusion_matrix

   # For each cell in the confusion matrix, add the corresponding ground truths
   # and predictions to the lists
   #for gt in range(len(cm)):
    #  for pred in range(len(cm)):
     #     y_true += [gt] * cm[gt][pred]
      #    y_pred += [pred] * cm[gt][pred]
   #print(
    #"Classification report rebuilt from confusion matrix:\n"
    #f"{metrics.classification_report(y_true, y_pred)}\n"
   #)
    
