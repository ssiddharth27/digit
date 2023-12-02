from flask import Flask, request
from joblib import dump, load
import numpy as np
from markupsafe import escape


app = Flask(__name__)



@app.route('/compare_images', methods=['POST'])
def compare_images():
    
    js = request.get_json()
    image1 = js['image1']
    image2 = js['image2']
    print("hello ")
    
    img1 = np.array([float(i) for i in image1]).reshape(1,-1)
    img2 = np.array([float(i) for i in image2]).reshape(1,-1)
    
    model = load('./models/svm_gamma:0.001_C:0.1.joblib')
    prediction1 = model.predict(img1)
    prediction2 = model.predict(img2)

    # Compare the predicted digits
    if np.argmax(prediction1) == np.argmax(prediction2):
        return "True"
    else:
        return "False"
        
@app.route('/predict/<model_type>', methods=['POST'])

def predict():
     
     
    model_type = f'predict {escape(model_type)}'
    if model_type == "svm":
       model = load('./models/svm_gamma:0.001_C:0.1.joblib')
    elif model_type == "tree":
       model = load("./models/tree_max_depth:20.joblib")
    else:
       model = load("./final_exam_models/B20EE067_lr_liblinear.joblib")
       
    js = request.get_json()
    image1 = js['image1']
       
    prediction = model.predict(image1)[0]  # Assuming data is a list or array
    
    # Return the prediction as JSON
    return str(int(prediction))
       
    
  
