import pickle
import pandas as pd
import numpy as np

def load_model(model_path='../models/best_model.pkl'):
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def predict_churn(data, model_data):
    """
    Make churn predictions for new customer data
    
    Parameters:
    data (dict): Dictionary containing customer features
    model_data (dict): Dictionary containing model and feature names
    
    Returns:
    dict: Prediction results including probability
    """
    # Convert input data to DataFrame
    df = pd.DataFrame([data])
    
    # Ensure all features are present
    for feature in model_data['feature_names']:
        if feature not in df.columns:
            df[feature] = 0
    
    # Reorder columns to match training data
    df = df[model_data['feature_names']]
    
    # Make prediction
    probability = model_data['model'].predict_proba(df)[0][1]
    prediction = 'Yes' if probability >= 0.5 else 'No'
    
    return {
        'churn_prediction': prediction,
        'churn_probability': float(probability),
        'confidence': float(abs(probability - 0.5) * 2)
    } 