from api.app import app
import json
from sklearn import datasets, metrics, svm
import pytest

def test_get_root():
	response = app.test_client().get("/")
	assert response.status_code == 200
	assert response.get_data() == b"<p>Hello, World!</p>"

def test_post_root():

	suffix = "post suffix"
	response = app.test_client().post("/", json={"suffix": suffix})
	assert response.status_code == 200    
	assert response.get_json()['op'] == "Hello, World POST " + suffix

def test_post_predict():

        digit_samples = []
        digits = datasets.load_digits()

        
        for i in range(10):
        
            digit_indices = (digits.target == i)
            digit_sample = digits.data[digit_indices][0]
            digit_samples.append(list(digit_sample))
        
        
        sample_payloads = [{"data": digit_samples[0]}, {"data": digit_samples[1]}, {"data": digit_samples[2]},
	                   {"data": digit_samples[3]}, {"data": digit_samples[4]}, {"data": digit_samples[5]},
	                   {"data": digit_samples[6]}, {"data": digit_samples[7]}, {"data": digit_samples[8]},
	                   {"data": digit_samples[9]}]
	                   
        for digit, payload in enumerate(sample_payloads):
            
            with app.test_client() as client:
            
                
                response = client.post("/predict",json=payload)
                #assert response.status_code == 200
                
                        # Parse the JSON response and extract the predicted digit
                response_data = json.loads(response.get_data(as_text=True))
                predicted_digit = response_data['prediction']
                assert predicted_digit == digit

        # Assert predicted digit for each payload
        #assert predicted_digit == digit
         #       assert int(response.get_data(as_text = True)) == digit
                assert response.status_code == 200
	



