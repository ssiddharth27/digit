from flask import Flask, request
from joblib import dump, load
import numpy as np

app = Flask(__name__)

model = load('/home/siddharth/Documents/Ml-ops/Digit_classification/digit/models/svm_gamma:0.001_C:1.joblib')


@app.route('/compare_images', methods=['POST'])
def compare_images():
    
    js = request.get_json()
    image1 = js['image1']
    image2 = js['image2']

    prediction1 = model.predict(image1)
    prediction2 = model.predict(image2)

    # Compare the predicted digits
    if np.argmax(prediction1) == np.argmax(prediction2):
        return True
    else:
        return False
