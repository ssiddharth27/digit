from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/", methods=["POST"])
def hello_world_post():    
    return {"op" : "Hello, World POST " + request.json["suffix"]}
    
@app.route('/predict', methods=['POST'])
def predict():
    # Your prediction logic goes here
    # This is just a placeholder response
    return jsonify({'prediction': 0})
