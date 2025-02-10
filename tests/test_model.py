import pytest
from src.models.predict import predict_churn, load_model

def test_model_loading():
    model_data = load_model()
    assert model_data is not None
    assert 'model' in model_data
    assert 'feature_names' in model_data

def test_prediction():
    model_data = load_model()
    test_data = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'DSL',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'No',
        'DeviceProtection': 'Yes',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 70.35,
        'TotalCharges': 1010.25
    }
    
    result = predict_churn(test_data, model_data)
    assert 'churn_prediction' in result
    assert 'churn_probability' in result
    assert isinstance(result['churn_probability'], float)
    assert 0 <= result['churn_probability'] <= 1 