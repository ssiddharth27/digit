from sklearn import datasets, metrics, svm, tree
from sklearn.model_selection import train_test_split
from joblib import dump, load

def preprocess_data(data):
    # flatten the images
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data
   
def train_model(x, y, model_params, model_type = "svm"):
   if model_type == "svm":
      clf = svm.SVC
   if model_type == "tree":
      clf = tree.DecisionTreeClassifier
   model = clf(**model_params)
   model.fit(x,y)
   return model

def split_train_dev_test(X, y, test_size, dev_size):

   # calculating the size of training data
   train_size = 1.0 - test_size - dev_size

   X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size= test_size,
   shuffle = False, random_state = 42)
   
   new_dev_size = dev_size / (dev_size + train_size)
   
   X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size= new_dev_size,
   shuffle = False, random_state = 42)
   
   return X_train, X_test, y_train, y_test, X_val, y_val
   
def get_combo(param_name, param_values, base_combinations):  
  
    new_combinations = []
    for value in param_values:
        for combination in base_combinations:
            combination[param_name] = value
            new_combinations.append(combination.copy()) 
               
    return new_combinations

def get_hyperpara_combo(dict_of_param_lists): 
   
    base_combinations = [{}]
    for param_name, param_values in dict_of_param_lists.items():
        base_combinations = get_combo(param_name, param_values, base_combinations)
        
    return base_combinations
   
def predict_and_eval(model, X_test, y_test):

   # prediction
   predicted = model.predict(X_test)
   
   accuracy = metrics.accuracy_score(y_test, predicted)
   
   return accuracy
   
def tune_hparams(X_train, Y_train, X_dev, y_dev, list_of_all_param_combinations, model_type):

   best_acc = -1  
   best_model_path = ""
   
   for params in list_of_all_param_combinations:

        #hyperparameter_settings = dict(zip(hyperparameters.keys(), params))

        #cur_gamma = hyperparameter_settings['gamma']
        #cur_crange = hyperparameter_settings['c_ranges']
	
        cur_model = train_model(X_train, Y_train, params, model_type)
        #cur_model.fit(X_train, Y_train)

        cur_dev_acc = predict_and_eval(cur_model, X_dev, y_dev)
        if cur_dev_acc > best_acc:
            best_acc = cur_dev_acc
            best_hparams = params
            best_model_path = "/digit/models/{}_".format(model_type) + "_".join(["{}:{}".format(k,v) for k,v in params.items()]) +   ".joblib"
            print(best_model_path)                   
            best_model = cur_model
            
        dump(best_model, best_model_path) 


   return best_hparams, best_model_path, best_acc
	    
   
   
#def tune_hparams_dt(X_train, Y_train, X_dev, y_dev, hyperparameters, list_of_all_param_combinations):

   #best_acc = -1  
   #hyperparameters_dt = {'depth':[None,10,20,30],"sample_split":[2,5,10]}
   
   #for params in list_of_all_param_combinations:

    #    hyperparameter_settings = dict(zip(hyperparameters.keys(), params))

     #   depth = hyperparameter_settings['depth']
     #   sample_split = hyperparameter_settings['sample_split']
	
      #  cur_model = DecisionTreeClassifier(max_depth=depth,min_samples_split = sample_split)
       # cur_model.fit(X_train, Y_train)

       # cur_dev_acc = predict_and_eval(cur_model, X_dev, y_dev)
       # if cur_dev_acc > best_acc:
        #    best_acc = cur_dev_acc
         #   best_gamma = depth
          #  best_crange = sample_split
           # best_model = cur_model
	    
  # return best_gamma, best_crange, best_model, best_acc
	    
	 

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
    
