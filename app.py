from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the trained RandomForest model
with open('rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the scaler used during training
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Render the home page
@app.route('/')
def home():
    return render_template('index.html')

# Handle the form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    material_quantity = float(request.form['material_quantity'])
    additive_catalyst = float(request.form['additive_catalyst'])
    ash_component = float(request.form['ash_component'])
    water_mix = float(request.form['water_mix'])
    plasticizer = float(request.form['plasticizer'])
    moderate_aggregator = float(request.form['moderate_aggregator'])
    refined_aggregator = float(request.form['refined_aggregator'])
    formulation_duration = float(request.form['formulation_duration'])

    # Scale the input data using the loaded scaler
    input_data = np.array([[material_quantity, additive_catalyst, ash_component, water_mix, plasticizer,
                            moderate_aggregator, refined_aggregator, formulation_duration]])
    scaled_input_data = scaler.transform(input_data)

    # Make a prediction
    prediction = model.predict(scaled_input_data)

    # Display the prediction on the result page
    return render_template('result.html', prediction=f'The predicted compressive strength is {prediction[0]:.2f} MPa.',
                           input_data=f'Input Data: {input_data}')

if __name__ == '__main__':
    app.run(debug=True)
