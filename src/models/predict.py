import pickle
import pandas as pd
import numpy as np
import os

def load_model(model_path=None):
    """Load the trained model"""
    if model_path is None:
        # Get the absolute path to the models directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        model_path = os.path.join(project_root, 'models', 'best_model.pkl')
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def predict_churn(data, model_data):
    """
    Make churn predictions for new customer data
    """
    try:
        # Create DataFrame from input data
        df = pd.DataFrame([data])
        
        # Encode categorical variables
        for feature in model_data['categorical_features']:
            if feature in df.columns:
                encoder = model_data['encoders'][feature]
                df[feature] = encoder.transform(df[feature].astype(str))
        
        # Scale numerical features
        if model_data['numerical_features']:
            df[model_data['numerical_features']] = model_data['scaler'].transform(
                df[model_data['numerical_features']]
            )
        
        # Ensure correct feature order
        df = df[model_data['feature_names']]
        
        # Make prediction
        probability = model_data['model'].predict_proba(df)[0][1]
        prediction = 'Yes' if probability >= 0.5 else 'No'
        
        return {
            'churn_prediction': prediction,
            'churn_probability': float(probability),
            'confidence': float(abs(probability - 0.5) * 2)
        }
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return {
            'error': str(e),
            'churn_prediction': 'Error',
            'churn_probability': 0.0,
            'confidence': 0.0
        } 