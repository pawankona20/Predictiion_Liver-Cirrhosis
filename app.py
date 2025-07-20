from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

try:
    with open('gaussiannb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    imputer = None
    onehotencoder = None
    categorical_cols = []
    training_columns = []

except FileNotFoundError:
    model = None
    imputer = None
    onehotencoder = None
    categorical_cols = []
    training_columns = []
    print("Error: model, imputer, onehotencoder, or categorical_cols file not found. The application will not work correctly.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model or preprocessing components not loaded. Cannot make predictions.", 500

    try:
        form_data = request.form.to_dict()
        input_data = [float(request.form.get(f'feature{i+1}', np.nan)) for i in range(4)] # Adjust range based on number of features
        features = np.array([input_data])
        prediction = model.predict(features)[0]
        prediction_label = "No Cirrhosis" if prediction == 0 else "Cirrhosis"
    except Exception as e:
        return f"Error processing input or making prediction: {e}", 400

    return render_template('result.html', prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
