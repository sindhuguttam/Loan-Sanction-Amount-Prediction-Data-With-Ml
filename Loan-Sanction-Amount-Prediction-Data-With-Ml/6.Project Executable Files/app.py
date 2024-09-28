from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the pre-trained pipeline which includes the preprocessor and the regressor
pipeline = joblib.load('pipeline.pkl')

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    form_data = request.form.to_dict()
    
    # Convert form data to DataFrame
    input_data = pd.DataFrame([form_data])
    
    # Cast numeric columns to float
    numeric_columns = ['Age', 'Income (USD)', 'Loan Amount Request (USD)', 'Current Loan Expenses (USD)', 
                       'Dependents', 'Credit Score', 'Property Age', 'Property Price']
    for column in numeric_columns:
        input_data[column] = pd.to_numeric(input_data[column], errors='coerce')
    
    # Print the form data and preprocessed data for debugging
    print("Form Data:", form_data)
    print("Input Data for Prediction:", input_data)
    
    # Use the pipeline to preprocess and predict
    try:
        loan_eligibility = pipeline.predict(input_data)[0]
        print("Prediction:", loan_eligibility)
        return render_template('result.html', prediction=loan_eligibility)
    except ValueError as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
