import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

def train_and_save_model():
    # Get absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_path = os.path.join(project_root, 'data', 'Telco_Customer_Churn.csv')
    models_dir = os.path.join(project_root, 'models')
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Convert TotalCharges to numeric and handle missing values properly
    df = df.copy()  # Create a copy to avoid the warning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.loc[df['TotalCharges'].isna(), 'TotalCharges'] = df.loc[df['TotalCharges'].isna(), 'MonthlyCharges']
    
    # Prepare features and target
    features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
               'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
               'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
               'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
               'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
    
    X = df[features].copy()
    y = (df['Churn'] == 'Yes').astype(int)
    
    # Encode categorical variables
    categorical_features = X.select_dtypes(include=['object']).columns
    encoders = {}
    
    for feature in categorical_features:
        le = LabelEncoder()
        X[feature] = le.fit_transform(X[feature].astype(str))
        encoders[feature] = le
    
    # Scale numerical features
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model and preprocessing objects
    model_data = {
        'model': model,
        'encoders': encoders,
        'scaler': scaler,
        'feature_names': features,
        'numerical_features': numerical_features,
        'categorical_features': list(categorical_features)
    }
    
    model_path = os.path.join(models_dir, 'best_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model trained and saved to {model_path}")
    return model_data

if __name__ == "__main__":
    train_and_save_model() 