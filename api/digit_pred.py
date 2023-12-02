from flask import Flask, request
from joblib import dump, load
import numpy as np


app = Flask(__name__)



@app.route('/compare_images', methods=['POST'])
def compare_images():
    
    js = request.get_json()
    image1 = js['image1']
    image2 = js['image2']
    print("hello ")
    
    img1 = np.array([float(i) for i in image1]).reshape(1,-1)
    img2 = np.array([float(i) for i in image2]).reshape(1,-1)
    
    model = load('/home/siddharth/Documents/Ml-ops/Digit_classification/digit/models/svm_gamma:0.001_C:1.joblib')
    prediction1 = model.predict(img1)
    prediction2 = model.predict(img2)

    # Compare the predicted digits
    if np.argmax(prediction1) == np.argmax(prediction2):
        return "True"
    else:
        return "False"
