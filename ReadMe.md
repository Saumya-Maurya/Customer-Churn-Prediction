# Telco Customer Churn Prediction

## 📊 Project Overview
A machine learning solution for predicting customer churn in the telecommunications industry. This project includes data analysis, model development, and a web-based dashboard for real-time predictions.

## 🌟 Features
- Interactive dashboard for data visualization
- Real-time churn prediction
- Comprehensive data analysis
- Machine learning model with RandomForest classifier
- REST API for predictions
- Automated CI/CD pipeline

## 🛠️ Technology Stack
- Python 3.8+
- Streamlit
- Scikit-learn
- Pandas
- Plotly
- FastAPI (for API)
- GitHub Actions (CI/CD)

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/telco-churn-prediction.git
cd telco-churn-prediction
```

2. Create and activate virtual environment:

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Unix/Mac
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:

```bash
streamlit run streamlit_app/app.py
```

## 🚀 Project Structure

telco-churn-prediction/
├── data/ # Dataset storage
├── notebooks/ # Jupyter notebooks for analysis
├── src/ # Source code
│ ├── preprocessing/ # Data preprocessing modules
│ ├── models/ # ML model implementations
│ └── api/ # API implementation
├── tests/ # Unit tests
├── streamlit_app/ # Streamlit dashboard
├── docs/ # Documentation
└── config/ # Configuration files

## 📈 Model Performance
- Accuracy: 0.82
- Precision: 0.76
- Recall: 0.73
- F1 Score: 0.74

## 🔄 CI/CD Pipeline
Our CI/CD pipeline automates:
- Code quality checks
- Unit tests
- Model training validation
- Deployment to production

## 📝 Documentation
Detailed documentation is available in the [docs](./docs) directory:
- [API Documentation](./docs/api.md)
- [Model Documentation](./docs/model.md)
- [Dashboard Guide](./docs/dashboard.md)

## 🚀 Deployment
See [Deployment Guide](./docs/deployment.md) for detailed instructions.

## 🤝 Contributing
Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md).

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
