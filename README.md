# Medical Insurance Cost Prediction 

An end-to-end machine learning pipeline that accurately predicts medical insurance costs based on demographic and lifestyle factors. This project demonstrates full MLOps implementation with automated training, tracking, and deployment.

##  What This Project Does

This system predicts individual medical insurance charges using machine learning, helping insurance companies and consumers estimate healthcare costs more accurately.

###  Key Features

- ** Accurate Cost Prediction** - Predicts insurance premiums with 87%+ accuracy using ensemble methods
- ** Full MLOps Pipeline** - Automated data ingestion, transformation, and model training
- ** MLflow Integration** - Comprehensive experiment tracking and model management
- ** DVC Version Control** - Data and model versioning for reproducibility
- ** CI/CD Deployment** - Automated testing and deployment with GitHub Actions
- ** Web Interface** - Flask-based web application for real-time predictions
- ** Containerized** - Docker support for easy deployment

##  Tech Stack

- **Machine Learning**: Scikit-learn, XGBoost, CatBoost, Random Forest
- **MLOps**: MLflow, DVC, Git
- **Backend**: Flask, Python
- **Deployment**: Docker, GitHub Actions, Render
- **Data Processing**: Pandas, NumPy, Scikit-learn preprocessing
- **Tracking**: DagsHub MLflow

##  Prerequisites

- Python 3.8+
- Git
- DVC (optional, for data versioning)
- MLflow account (or DagsHub for tracking)

##  Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Atanu-Manna2003/Medical-Insurance-Cost-Prediction.git
cd medical-insurance-cost-prediction

python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt

#4. Set Up Environment Variables
MLFLOW_TRACKING_URI=https://dagshub.com/yourusername/your-repo.mlflow
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_token

set MLFLOW_TRACKING_URI=""

set MLFLOW_TRACKING_USERNAME=""
set MLFLOW_TRACKING_PASSWORD=""

run python main.py
Visit http://localhost:5000 to access the prediction interface.