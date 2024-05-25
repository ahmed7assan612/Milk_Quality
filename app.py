import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

flask_app = Flask(name)
model = pickle.load(open("model_milk.pkl", "rb"))

@flask_app.route("/")
def welcome():
    return "Welcome All"

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template(prediction_text = "The Answer {}".format(prediction))

if __name__ == "main":
    flask_app.run(debug=True)