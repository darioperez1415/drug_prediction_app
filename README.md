# Drug Prediction App

A machine learning application to predict drug usage behavior based on demographic and personality traits.

Built with:
- Python 🐍
- Random Forest and Decision Tree Classifiers 🌳
- Streamlit 📈

---

## 🚀 Project Overview

This application analyzes user data and predicts the likelihood of drug consumption for various substances (e.g., alcohol, cannabis, caffeine, etc.).

Key Features:
- Interactive web app using Streamlit
- Trained models (Random Forest, Decision Tree) with SMOTE balancing
- Performance metrics: Accuracy, Precision, Recall, F1 Score
- Easy deployment ready!

---

## 📁 Project Structure

```bash
├── app/
│   ├── app.py              # Streamlit application
│   ├── model_training.py   # Model training scripts
├── data/
│   └── drug_consumption_combined.csv  # Dataset
├── requirements.txt        # Project dependencies
└── README.md                # Project documentation
