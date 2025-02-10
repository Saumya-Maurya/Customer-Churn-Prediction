import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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
    
    print("Loading data...")
    # Load data
    df = pd.read_csv(data_path)
    
    # Convert TotalCharges to numeric and handle missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)
    
    print("Preparing features...")
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
    
    print("Splitting data...")
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Calculating metrics...")
    # Calculate metrics
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        'Training_Accuracy': float(accuracy_score(y_train, y_pred_train)),
        'Test_Accuracy': float(accuracy_score(y_test, y_pred_test)),
        'Precision': float(precision_score(y_test, y_pred_test)),
        'Recall': float(recall_score(y_test, y_pred_test)),
        'F1_Score': float(f1_score(y_test, y_pred_test))
    }
    
    # Calculate and sort feature importance
    feature_importance = dict(zip(features, model.feature_importances_))
    feature_importance = {k: float(v) for k, v in feature_importance.items()}
    
    print("Saving model...")
    # Save model and preprocessing objects
    model_data = {
        'model': model,
        'encoders': encoders,
        'scaler': scaler,
        'feature_names': features,
        'numerical_features': numerical_features,
        'categorical_features': list(categorical_features),
        'metrics': metrics,
        'feature_importance': feature_importance,
        'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()
    }
    
    model_path = os.path.join(models_dir, 'best_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model training completed!")
    print("\nModel Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return model_data

if __name__ == "__main__":
    train_and_save_model() 