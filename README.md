<img width="1021" height="657" alt="image" src="https://github.com/user-attachments/assets/48ed2384-f828-4485-a161-36503454adf9" /># 🩺 Diabetes Prediction ML Project

An end-to-end Machine Learning project to predict diabetes 
using multiple classification algorithms.

## 🚀 Live Demo
- 🎯 Streamlit: https://diabetes-prediction-mythili.streamlit.app/
- 🤗 HuggingFace: https://huggingface.co/spaces/Mythili-Pandiyarajan/diabetes-prediction


## 🔍 Project Overview
This project compares 4 ML models:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* XGBoost Classifier

## 📁 Project Structure

| File | Description |
|------|-------------|
| `diabetes_prediction.ipynb` | ML notebook with EDA and model training |
| `app.py` | Streamlit web application |
| `diabetes_model.pkl` | Trained ML model |
| `diabetes.csv` | Dataset used |
| `requirements.txt` | Required libraries |

## 🛠️ Libraries Used

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Matplotlib, Seaborn

## 🚀 How to Run Locally

```bash
git clone https://github.com/Mythili-Pandiyarajan/diabetes-prediction-ml.git
cd diabetes-prediction-ml
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Model Performance

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------| ---------- |
| Logistic Regression | 0.7135 | 0.5985 | 0.7337 |
| Decision Tree | 0.7031 | 0.5649 | 0.7940 |
| Random Forest | 0.7656 | 0.6897 | 0.8214 |
| XGBoost | 0.7552 | 0.6667 | 0.8102 |

> ✅ Best Model: Random Forest with ROC-AUC of 0.8214
> 
## 📌 Dataset
Dataset sourced from Kaggle - Pima Indians Diabetes Dataset

## 👩‍💻 Author
Mythili Pandiyarajan __[GitHub Profile](https://github.com/Mythili-Pandiyarajan)__
