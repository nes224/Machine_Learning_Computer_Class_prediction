import numpy as np 
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
# Create Flask app
app = Flask(__name__)
df = pd.read_csv("computer_class.csv")
# Load the pickle model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    if prediction[0] == 'computer_office':
        filtered_df = df[df["Computer_length"] == 1]
        
    else:
        filtered_df = df[df["Computer_length"] == 0]
    return render_template("index.html", prediction_text = "The flower species is {}".format(filtered_df.to_json(orient="records")))


if __name__ == "__main__":
    app.run(debug=True)