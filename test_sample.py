from utils import get_hyperpara_combo
    
from sklearn.base import is_classifier
from sklearn.linear_model import LogisticRegression

    
def test_hyper_count():
    
    gamma_list = [0.001,0.01,0.1,1]
    c_val = [1,10,100,1000]
    hyper_para = {}
    hyper_para['gamma'] = gamma_list
    hyper_para['c_value'] = c_val
    
    all_combination = get_hyperpara_combo(hyper_para)
    
    assert len(all_combination) == len(gamma_list) * len(c_val)
    
def test_hyper_value():

    gamma_list = [0.001,0.01,0.1,1]
    c_val = [1,10,100,1000]
    hyper_para = {}
    hyper_para['gamma'] = gamma_list
    hyper_para['c_value'] = c_val
    
    all_combination = get_hyperpara_combo(hyper_para)
    expected_1 = {'c_value': 1, "gamma":0.1}
    expected_2 = {'c_value': 100, "gamma":0.01}
    
    assert (expected_1 in all_combination)  and (expected_2 in all_combination)
    
def check_model():
     
    model_name = "/home/siddharth/Documents/Ml-ops/Digit_classification/digit/final_exam_models/B20EE067_lr_lbfgs.joblib"
   # model_filename = f"{roll_number}_lr_{solver}.joblib"
    model = joblib.load(model_name)
    assert is_classifier(model) and isinstance(model, LogisticRegression), f"Loaded model for {solver} is not a logistic regression model."
    loaded_solver = model.get_params()['solver']
    assert solver == loaded_solver, f"Solver name in the filename does not match the solver used in the model."

    print(f"Tests passed for model: {model_filename}")
    print()
    


