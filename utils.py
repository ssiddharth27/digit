from sklearn import metrics
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
   
   # metrics
   print(
    f"Classification report for classifier {model}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
    ) 
    
   disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
   disp.figure_.suptitle("Confusion Matrix")
   print(f"Confusion matrix:\n{disp.confusion_matrix}")

   # The ground truth and predicted lists
   y_true = []
   y_pred = []
   cm = disp.confusion_matrix

   # For each cell in the confusion matrix, add the corresponding ground truths
   # and predictions to the lists
   for gt in range(len(cm)):
      for pred in range(len(cm)):
          y_true += [gt] * cm[gt][pred]
          y_pred += [pred] * cm[gt][pred]
   print(
    "Classification report rebuilt from confusion matrix:\n"
    f"{metrics.classification_report(y_true, y_pred)}\n"
   )
    
