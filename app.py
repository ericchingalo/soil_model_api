import flask
from flask import Flask, request, jsonify
import sys
import numpy as np
import pandas as pd
from model.load import * 
from sklearn.preprocessing import StandardScaler
import pickle


global  model


def create_app(config_filename):
    app = Flask(__name__)
    app.config.from_object(config_filename)

    return app

# model = init()
model = pickle.load(open('new_maize.pkl', 'rb'))
app = create_app("config")

@app.route('/soil-model/predict/',methods=['POST'])
def predict():
    if flask.request.method == 'POST':
    
        json_data = request.get_json(force=True)
        if not json_data:
            return {'message': 'No input data provided'}, 400
        
        temperature = json_data['temperature']
        moisture = json_data['moisture']
        pH = json_data['pH']

        input_variables = pd.DataFrame([[moisture, temperature, pH]], dtype=float)
        # sc_X = StandardScaler()
        # inputs = sc_X.fit_transform(input_variables)
        prediction = model.predict(input_variables)

        # prob = prediction.argmax(axis=-1)
        # print(inputs)
        # print(prob[0])
        # print(prediction[0])
        if prediction[0] > 0: 
            maize = True
        else:
            maize = False

    response = "For ML Prediction"
    return {"maize":  maize}	


if __name__ == "__main__":
    app.run(debug=True, port=3007)
