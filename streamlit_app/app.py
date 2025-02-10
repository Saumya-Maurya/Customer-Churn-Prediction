import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.models.predict import load_model, predict_churn

# Load the model and data
model_data = load_model()  # No need to specify path, it will use the default
df = pd.read_csv(os.path.join(project_root, 'data', 'Telco_Customer_Churn.csv'))

# Page config
st.set_page_config(page_title="Telco Customer Churn Analysis", page_icon="📊", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", 
    ["Dashboard Overview", 
     "Customer Demographics", 
     "Services Analysis",
     "Financial Analysis",
     "Churn Prediction"])

if page == "Dashboard Overview":
    st.title("📊 Telco Customer Churn Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_customers = len(df)
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col2:
        churn_rate = (df['Churn'] == 'Yes').mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    with col3:
        avg_tenure = df['tenure'].mean()
        st.metric("Avg. Tenure (months)", f"{avg_tenure:.1f}")
    
    with col4:
        avg_monthly = df['MonthlyCharges'].mean()
        st.metric("Avg. Monthly Charges", f"${avg_monthly:.2f}")
    
    # Monthly Revenue Trend
    st.subheader("Monthly Revenue Analysis")
    monthly_revenue = df.groupby('tenure')['MonthlyCharges'].sum().reset_index()
    fig = px.line(monthly_revenue, x='tenure', y='MonthlyCharges',
                  title='Monthly Revenue Trend by Tenure')
    st.plotly_chart(fig, use_container_width=True)
    
    # Churn by Contract Type
    col1, col2 = st.columns(2)
    with col1:
        contract_churn = df.groupby(['Contract', 'Churn']).size().unstack()
        fig = px.bar(contract_churn, title='Churn by Contract Type',
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        service_counts = df.groupby('InternetService')['Churn'].value_counts().unstack()
        fig = px.bar(service_counts, title='Churn by Internet Service',
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)

elif page == "Customer Demographics":
    st.title("👥 Customer Demographics Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gender Distribution
        gender_churn = df.groupby(['gender', 'Churn']).size().unstack()
        fig = px.bar(gender_churn, title='Gender Distribution by Churn Status',
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        # Senior Citizen Analysis
        senior_churn = df.groupby(['SeniorCitizen', 'Churn']).size().unstack()
        fig = px.bar(senior_churn, title='Senior Citizen Distribution by Churn Status',
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Partner Status
        partner_churn = df.groupby(['Partner', 'Churn']).size().unstack()
        fig = px.bar(partner_churn, title='Partner Status by Churn',
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        # Dependents Status
        dependents_churn = df.groupby(['Dependents', 'Churn']).size().unstack()
        fig = px.bar(dependents_churn, title='Dependents Status by Churn',
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)

elif page == "Services Analysis":
    st.title("🔧 Services Analysis")
    
    # Service adoption rates
    services = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    service_adoption = pd.DataFrame()
    for service in services:
        service_adoption[service] = df[service].value_counts(normalize=True)
    
    fig = px.bar(service_adoption.T, title='Service Adoption Rates',
                 barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    
    # Service combinations
    col1, col2 = st.columns(2)
    
    with col1:
        internet_phone = df.groupby(['InternetService', 'PhoneService']).size().unstack()
        fig = px.bar(internet_phone, title='Internet and Phone Service Combinations',
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        streaming = df.groupby(['StreamingTV', 'StreamingMovies']).size().unstack()
        fig = px.bar(streaming, title='Streaming Services Combinations',
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)

elif page == "Financial Analysis":
    st.title("💰 Financial Analysis")
    
    # Convert TotalCharges to numeric and handle missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)
    
    # Monthly Charges Distribution
    fig = px.histogram(df, x='MonthlyCharges', 
                      title='Distribution of Monthly Charges',
                      color='Churn')
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Average Monthly Charges by Contract
        avg_monthly = df.groupby('Contract')['MonthlyCharges'].mean().reset_index()
        fig = px.bar(avg_monthly, x='Contract', y='MonthlyCharges',
                    title='Average Monthly Charges by Contract Type')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Payment Method Analysis
        payment_churn = df.groupby(['PaymentMethod', 'Churn']).size().unstack()
        fig = px.bar(payment_churn, title='Payment Methods by Churn Status',
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    # Tenure vs Charges Analysis with fixed size values
    fig = px.scatter(df, 
                    x='tenure', 
                    y='MonthlyCharges',
                    color='Churn',
                    size=[20] * len(df),  # Fixed size instead of using TotalCharges
                    title='Tenure vs Monthly Charges',
                    labels={
                        'tenure': 'Tenure (months)',
                        'MonthlyCharges': 'Monthly Charges ($)'
                    })
    
    # Update layout
    fig.update_layout(
        showlegend=True,
        xaxis_title="Tenure (months)",
        yaxis_title="Monthly Charges ($)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional financial metrics
    st.subheader("Financial Metrics Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_monthly = df['MonthlyCharges'].mean()
        st.metric("Average Monthly Charges", f"${avg_monthly:.2f}")
    
    with col2:
        avg_total = df['TotalCharges'].mean()
        st.metric("Average Total Charges", f"${avg_total:.2f}")
    
    with col3:
        revenue_risk = df[df['Churn'] == 'Yes']['MonthlyCharges'].sum()
        st.metric("Monthly Revenue at Risk", f"${revenue_risk:.2f}")

else:  # Churn Prediction page
    st.title("🔮 Churn Prediction")
    
    # Input fields
    with st.sidebar:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.checkbox("Senior Citizen")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        
        st.subheader("Services")
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        
        if internet_service != "No":
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        else:
            online_security = online_backup = device_protection = tech_support = "No internet service"
            streaming_tv = streaming_movies = "No internet service"
        
        st.subheader("Contract Details")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", 
                                    ["Electronic check", "Mailed check", 
                                     "Bank transfer (automatic)", 
                                     "Credit card (automatic)"])
        
        st.subheader("Financial Information")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 20, 120, 50)
        total_charges = monthly_charges * tenure  # Calculate total charges

    # Create prediction button
    if st.sidebar.button("Predict Churn"):
        # Prepare input data
        input_data = {
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': float(tenure),
            'PhoneService': phone_service,
            'MultipleLines': 'No' if phone_service == 'No' else 'Yes',
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': float(monthly_charges),
            'TotalCharges': float(total_charges)
        }
        
        # Make prediction
        result = predict_churn(input_data, model_data)
        
        if 'error' in result:
            st.error(f"An error occurred: {result['error']}")
        else:
            # Display results in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Churn Prediction",
                    "Will Churn" if result['churn_prediction'] == 'Yes' else "Will Stay"
                )
            
            with col2:
                st.metric(
                    "Churn Probability",
                    f"{result['churn_probability']:.1%}"
                )
            
            with col3:
                st.metric(
                    "Prediction Confidence",
                    f"{result['confidence']:.1%}"
                )
            
            # Add gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = result['churn_probability'] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Risk"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            
            st.plotly_chart(fig)
            
            # Add risk factors
            st.subheader("Key Risk Factors")
            risk_factors = []
            
            if contract == "Month-to-month":
                risk_factors.append("Month-to-month contract")
            if internet_service == "Fiber optic":
                risk_factors.append("Fiber optic service")
            if payment_method == "Electronic check":
                risk_factors.append("Electronic check payment")
            if tenure < 12:
                risk_factors.append("Low tenure")
            if monthly_charges > 70:
                risk_factors.append("High monthly charges")
            
            if risk_factors:
                st.warning("Risk factors identified:\n" + "\n".join(f"• {factor}" for factor in risk_factors))
            else:
                st.success("No major risk factors identified")

    # Add information about the model
    with st.expander("Model Information"):
        st.subheader("Model Overview")
        st.write(f"Model Type: {type(model_data['model']).__name__}")
        
        try:
            if 'metrics' in model_data:
                st.subheader("Model Performance Metrics")
                
                # Create metrics columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Test Accuracy", 
                             f"{model_data['metrics']['Test_Accuracy']:.2%}")
                    st.metric("Training Accuracy", 
                             f"{model_data['metrics']['Training_Accuracy']:.2%}")
                
                with col2:
                    st.metric("Precision", 
                             f"{model_data['metrics']['Precision']:.2%}")
                    st.metric("Recall", 
                             f"{model_data['metrics']['Recall']:.2%}")
                
                with col3:
                    st.metric("F1 Score", 
                             f"{model_data['metrics']['F1_Score']:.2%}")
            
            if 'feature_importance' in model_data:
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': list(model_data['feature_importance'].keys()),
                    'Importance': list(model_data['feature_importance'].values())
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(importance_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title='Feature Importance')
                st.plotly_chart(fig, use_container_width=True)
            
            if 'confusion_matrix' in model_data:
                st.subheader("Confusion Matrix")
                cm = np.array(model_data['confusion_matrix'])
                
                # Create confusion matrix heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['No Churn', 'Churn'],
                    y=['No Churn', 'Churn'],
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 16},
                    hoverongaps=False,
                    colorscale='RdBu'
                ))
                
                fig.update_layout(
                    title='Confusion Matrix',
                    xaxis_title='Predicted',
                    yaxis_title='Actual',
                    width=400,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error displaying model information: {str(e)}")
            st.write("Model data:", model_data.keys())

    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit •Saumya Maurya Project") 