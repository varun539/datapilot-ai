📘 README.md
# 🚀 Varun's DataPilot AI

> **End-to-End AutoML Platform for Rapid Machine Learning Development**  
> Built by **Varun B**

DataPilot AI is a production-ready AutoML web application that allows users to upload datasets, automatically train ML models, analyze data, explain predictions, tune hyperparameters, manage model versions, and deploy predictions — all from a clean Streamlit interface.

This project demonstrates real-world machine learning engineering skills including feature pipelines, model lifecycle management, evaluation metrics, explainability, and deployment.

---

## 🌟 Key Features

### 📊 Data Profiling
- Automatic dataset inspection
- Missing value detection
- Data type analysis
- Health score & quality warnings

### 📈 Visual Analytics
- Numeric feature distributions
- Correlation heatmaps
- Categorical value counts
- Time-based trends (if applicable)

### 🤖 AutoML Engine
- Automatic problem detection (Regression / Classification)
- Multiple model training:
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - CatBoost
- Cross-validation scoring
- Automatic best-model selection

### ⚖️ Imbalanced Data Handling
- Detects class imbalance automatically
- Applies class weights when enabled
- Prevents biased models

### 📊 Model Evaluation (Classification)
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve (Binary classification)

### ⚡ Hyperparameter Optimization
- RandomizedSearchCV tuning
- Best parameter tracking
- Tuned model versioning

### 🧠 Explainability (SHAP)
- Global feature importance
- Model transparency
- Interpretability support

### 📦 Model Registry
- Automatic model versioning
- Track:
  - CV Score
  - Feature count
  - Hyperparameters
  - Timestamp
- Load any historical model version

### 🔮 Prediction Engine
- Single record prediction
- Batch CSV prediction
- Download prediction results

### 🌍 Deployment Ready
- Streamlit Cloud compatible
- Lightweight architecture
- Scalable structure

---

## 🏗️ Project Architecture



datapilot-ai/
│
├── app.py # Main Streamlit application
├── src/
│ ├── automl.py # Training, tuning, imbalance handling
│ ├── pipeline.py # Feature engineering pipeline
│ ├── data_loader.py # CSV loading
│ ├── eda.py # Visualization utilities
│ ├── model_registry.py # Model version tracking
│ └── data_quality.py # Dataset health scoring
│
├── models/ # Saved trained models
├── requirements.txt
└── README.md


---

## 🚀 How to Run Locally

### 1️⃣ Clone Repository

git clone https://github.com/varun539/datapilot-ai.git
cd datapilot-ai

2️⃣ Create Environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run Application
streamlit run app.py


Open browser:

http://localhost:8501

☁️ Deploy on Streamlit Cloud

Push code to GitHub

Go to 👉 https://share.streamlit.io

Connect GitHub repo

Select:

app.py


Click Deploy 🚀

🎯 Example Use Cases

Kaggle dataset exploration

Startup MVP modeling

College ML projects

Rapid prototyping

AutoML pipelines

Model comparison

Deployment demo projects

🧠 Tech Stack

Python

Streamlit

Scikit-learn

XGBoost

LightGBM

CatBoost

SHAP

Pandas / NumPy

Matplotlib

👨‍💻 Author

Varun B
Aspiring Machine Learning Engineer
Focused on building real-world AI systems 🚀

If you like this project — give it a ⭐ on GitHub!
