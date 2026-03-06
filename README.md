# 🚀 DataPilot AI — Agentic AutoML Platform

> **Upload your data. Train models. Get AI-powered insights. Chat with your results.**
> 
> Built by [Varun B](https://varun539.github.io/portfolio/) — Data Scientist & ML Engineer

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-Click_Here-blue?style=for-the-badge)](https://varun-datapilot-ai.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9+-green?style=for-the-badge&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![GPT-4o](https://img.shields.io/badge/GPT--4o-Powered-purple?style=for-the-badge&logo=openai)](https://openai.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

---

## 🤔 What is DataPilot AI?

Most AutoML tools give you a model and a number. **DataPilot AI gives you a conversation.**

Upload a CSV → DataPilot automatically trains the best ML model → GPT-4o explains results in plain English → Chat with your data to ask follow-up questions → Download a professional PDF report.

**No code needed. No data science degree needed. Just upload and understand.**

---

## ✨ What Makes This Different

| Feature | Typical AutoML Tool | DataPilot AI |
|---------|-------------------|--------------|
| Auto Model Training | ✅ | ✅ |
| SHAP Explainability | Sometimes | ✅ Always |
| GPT-4o Plain English Insights | ❌ | ✅ |
| Chat with Your Data | ❌ | ✅ Full conversation |
| AI Dataset Diagnosis | ❌ | ✅ |
| Smart Target Column Suggestion | ❌ | ✅ |
| PDF Business Report | ❌ | ✅ With AI narrative |
| Data Leakage Detection | ❌ | ✅ Auto-detected |
| Experiment Tracking | ❌ | ✅ |
| Model Registry | ❌ | ✅ |
| Premium Dark Theme Visuals | ❌ | ✅ |

---

## 🎬 How It Works — 4 Steps

```
1. UPLOAD     →  Drop your CSV
                 AI diagnoses dataset health instantly
                 AI suggests the best target column

2. TRAIN      →  5 models trained automatically
                 XGBoost, LightGBM, CatBoost,
                 Random Forest, Gradient Boosting
                 Best model selected via cross-validation

3. UNDERSTAND →  GPT-4o generates executive-level narrative
                 SHAP shows which features matter most
                 Plain English — zero jargon

4. ACT        →  Chat with your data
                 Download PDF report for your team
                 Run batch predictions on new data
```

---

## 🌟 Full Feature Breakdown

### 🤖 Agentic AI Layer — Powered by GPT-4o
- **AI Dataset Diagnosis** — tells you what's healthy and what needs fixing before training
- **Smart Target Suggester** — AI analyzes your columns and recommends the best prediction target
- **Executive Narrative** — GPT-4o writes a McKinsey-level analysis of your model results
- **Chat with Data** — ask anything about your dataset, model, or business insights
- **AI Prediction Explainer** — every prediction explained in plain English

### 📊 Data Intelligence
- Automatic dataset profiling with quality score (0–100)
- Smart ID column detection and auto-removal
- Duplicate and missing value detection
- Data leakage detection — stops training if suspicious correlations found

### 📈 Premium Visual Analytics
- Distribution plots with KDE curve and mean marker
- Half-triangle correlation heatmap with annotations
- Categorical bar charts with value labels
- Time series monthly + yearly trend charts
- All charts in premium dark theme

### 🏋️ AutoML Engine
- Auto-detects Regression vs Classification
- Auto-detects time-series vs standard ML
- Trains 5 models in parallel
- Smart feature pipeline — drops IDs, encodes categoricals, handles dates
- Class imbalance auto-detection and correction

### 🧠 Explainability (SHAP)
- Global feature importance visualization
- Top business drivers with direction analysis
- Business-safe language — no false causation claims

### ⚡ Hyperparameter Tuning
- RandomizedSearchCV optimization
- Best parameter tracking per run

### 📦 MLOps Features
- **Experiment Tracking** — every training run logged
- **Model Registry** — version control for trained models
- **Batch Prediction** — upload new CSVs, get predictions instantly
- **Confidence Intervals** — prediction ranges for regression

### 📄 PDF Report
- Professional business report with model overview
- GPT-4o executive narrative included
- Business impact + risk notices
- Ready to share with stakeholders

---

## 🏗️ Architecture

```
datapilot-ai/
│
├── app.py                       # Main Streamlit app — 10 pages
│
├── src/
│   ├── agent.py                 # GPT-4o Agentic AI layer
│   ├── automl.py                # Model training, tuning, imbalance handling
│   ├── pipeline.py              # Smart feature engineering pipeline
│   ├── eda.py                   # Premium dark theme visualizations
│   ├── data_loader.py           # CSV loading with encoding detection
│   ├── data_quality.py          # Dataset health scoring
│   ├── feature_engineering.py   # Advanced feature creation
│   ├── impact.py                # SHAP business insights
│   ├── report.py                # PDF report generation
│   ├── experiments.py           # Experiment logging
│   └── model_registry.py        # Model versioning
│
├── models/                      # Saved trained models
├── requirements.txt
└── README.md
```

> **Why modular architecture?** Each file has one job. Testable, maintainable, and production-ready — not a single notebook that breaks when you touch it.

---

## 💬 Chat with Data — Example

After training, you can ask anything:

**"What drives Profit the most?"**
> *Based on SHAP analysis, Sales is your top driver with importance score 0.847. Higher sales volumes are strongly associated with increased profit. Discount has a negative impact — every 10% increase in discount reduces expected profit by approximately 15%...*

**"Is this model good enough for production?"**
> *Your CatBoost model achieved R² of 0.83 with CV std of 0.05 — this is strong generalization. I'd recommend monitoring for data drift quarterly and retraining if R² drops below 0.75...*

**"What should the business do based on these results?"**
> *Three concrete actions: 1) Reduce discounts above 20% — they consistently destroy profit margins. 2) Focus on Technology sub-category — highest profit association. 3) Prioritize Corporate segment — outperforms Consumer by 23%...*

---

## 🚀 Quick Start

### Run Locally

```bash
# Clone
git clone https://github.com/varun539/datapilot-ai.git
cd datapilot-ai

# Install
pip install -r requirements.txt

# Add OpenAI key
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Run
streamlit run app.py
```

### Deploy on Streamlit Cloud

```
1. Fork this repo
2. Go to share.streamlit.io
3. Connect your GitHub repo → select app.py
4. Add secret: OPENAI_API_KEY = "sk-your-key"
5. Click Deploy 🚀
```

---

## 🎯 Example Results — Superstore Dataset

```
Dataset    : Sample Superstore (9,994 rows, 21 columns)
Target     : Profit prediction
Best Model : CatBoost
R² Score   : 0.83
Top Driver : Sales (SHAP: 0.847)
AI Insight : "Higher discounts are the #1 profit killer —
              reducing discounts above 20% could recover
              15-20% of lost margins"
```

---

## 🧠 Tech Stack

| Category | Tools |
|----------|-------|
| Frontend | Streamlit |
| ML Models | XGBoost, LightGBM, CatBoost, Scikit-learn |
| Explainability | SHAP |
| Agentic AI | OpenAI GPT-4o API |
| Data | Pandas, NumPy, SciPy |
| Visualization | Matplotlib, Seaborn |
| Reports | ReportLab |
| Deployment | Streamlit Cloud |
| Version Control | Git, GitHub |

---

## 🎯 Who Is This For?

| User | Use Case |
|------|----------|
| **Business Owners** | Understand sales, churn, or revenue without hiring a data scientist |
| **Data Scientists** | Rapid baseline model generation and prototyping |
| **Students** | Learn ML through a real production system |
| **Startups** | Validate data-driven hypotheses in minutes |
| **Analysts** | Go beyond Excel — get predictive models, not just charts |

---

## 👨‍💻 About the Author

**Varun B** — Data Scientist & ML Engineer from India

Building production-grade ML and AI systems — not just notebooks.

| | |
|--|--|
| 🌐 Portfolio | [varun539.github.io/portfolio](https://varun539.github.io/portfolio/) |
| 💼 LinkedIn | [linkedin.com/in/varun-b-647b1730a](https://www.linkedin.com/in/varun-b-647b1730a/) |
| 🐙 GitHub | [github.com/varun539](https://github.com/varun539) |
| 📧 Email | vb9225177@gmail.com |

*Open to remote Data Scientist / ML Engineer roles globally — available EU/UK overlap daily*

---

## ⭐ Support

If this project helped you or impressed you — give it a **star on GitHub!**

It helps more people find this project and supports open source ML tooling.

**Built with Python, SHAP, GPT-4o, and a lot of ☕ by Varun B**
