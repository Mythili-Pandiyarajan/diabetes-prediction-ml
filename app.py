import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
with open("diabetes_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
scaler = bundle["scaler"]

# Load dataset
df = pd.read_csv("diabetes.csv")

st.title("🩺 Diabetes Prediction App")

# ---- TABS ----
tab1, tab2 = st.tabs(["🔮 Predict", "📊 Data Insights"])

# ---- TAB 1: PREDICTION ----
with tab1:
    preg = st.number_input("Pregnancies", min_value=0, value=0)
    glucose = st.number_input("Glucose", min_value=0, value=120)
    bp = st.number_input("Blood Pressure", min_value=0, value=70)
    skin = st.number_input("Skin Thickness", min_value=0, value=20)
    insulin = st.number_input("Insulin", min_value=0, value=80)
    bmi = st.number_input("BMI", min_value=0.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
    age = st.number_input("Age", min_value=1, value=30)

    if st.button("Predict"):
        data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        log_idx = [3, 4, 6, 7]
        data[:, log_idx] = np.log1p(data[:, log_idx])
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)
        probability = model.predict_proba(data_scaled)[0][1]

        if prediction[0] == 1:
            st.error(f"⚠️ Diabetic ({probability:.2%} probability)")
        else:
            st.success(f"✅ Not Diabetic ({probability:.2%} probability)")

# ---- TAB 2: DATA INSIGHTS ----
with tab2:
    st.subheader("📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients", len(df))
    col2.metric("Diabetic", df['Outcome'].sum())
    col3.metric("Non-Diabetic", len(df) - df['Outcome'].sum())

    # Chart 1 - Diabetic vs Non-Diabetic
    st.subheader("🎯 Diabetic vs Non-Diabetic")
    fig1, ax1 = plt.subplots()
    df['Outcome'].value_counts().plot(
        kind='pie',
        labels=['Not Diabetic', 'Diabetic'],
        autopct='%1.1f%%',
        colors=['green', 'red'],
        ax=ax1
    )
    st.pyplot(fig1)

    # Chart 2 - Glucose Distribution
    st.subheader("🩸 Glucose Level Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(data=df, x='Glucose', hue='Outcome',
                kde=True, ax=ax2,
                palette={0: 'green', 1: 'red'})
    ax2.set_xlabel("Glucose Level")
    ax2.legend(["Not Diabetic", "Diabetic"])
    st.pyplot(fig2)

    # Chart 3 - BMI Distribution
    st.subheader("⚖️ BMI Distribution")
    fig3, ax3 = plt.subplots()
    sns.histplot(data=df, x='BMI', hue='Outcome',
                kde=True, ax=ax3,
                palette={0: 'green', 1: 'red'})
    ax3.set_xlabel("BMI")
    ax3.legend(["Not Diabetic", "Diabetic"])
    st.pyplot(fig3)

    # Chart 4 - Age Distribution
    st.subheader("👤 Age Distribution")
    fig4, ax4 = plt.subplots()
    sns.histplot(data=df, x='Age', hue='Outcome',
                kde=True, ax=ax4,
                palette={0: 'green', 1: 'red'})
    ax4.set_xlabel("Age")
    ax4.legend(["Not Diabetic", "Diabetic"])
    st.pyplot(fig4)

    # Chart 5 - Correlation Heatmap
    st.subheader("🔥 Feature Correlation")
    fig5, ax5 = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True,
               cmap='coolwarm', ax=ax5)
    st.pyplot(fig5)

    # Chart 6 - Average values by Outcome
    st.subheader("📈 Average Feature Values")
    fig6, ax6 = plt.subplots(figsize=(10, 5))
    df.groupby('Outcome').mean().T.plot(
        kind='bar', ax=ax6,
        color=['green', 'red']
    )
    ax6.legend(["Not Diabetic", "Diabetic"])
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45)
    st.pyplot(fig6)
