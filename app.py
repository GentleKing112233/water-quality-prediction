import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

st.set_page_config(page_title="Water Quality Prediction", layout="wide")
st.title("💧 Water Quality Prediction - Machine Learning Showcase")

st.markdown("""
This page showcases machine learning models for predicting water potability.
Upload the **water_potability.csv** dataset to explore the models and visualizations.
""")

uploaded_file = st.file_uploader("Upload water_potability.csv (optional)", type=["csv"])

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded_file is not None:
    data = load_data(uploaded_file)
else:
    try:
        data = load_data("water_potability.csv")
    except FileNotFoundError:
        st.info("Please upload the dataset to begin.")
        st.stop()

tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Visualizations", "Model Comparison"])

with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(data.head())
    st.subheader("Dataset Info")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Shape:", data.shape)
        st.write("Columns:", list(data.columns))
    with col2:
        st.write("Missing values:")
        st.write(data.isnull().sum())

with tab2:
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Potability Distribution")
    fig = px.pie(data, names="Potability", title="Potability Distribution", hole=0.4)
    fig.update_layout(title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Boxplots by Feature")
    fig, axs = plt.subplots(ncols=5, nrows=2, figsize=(20, 10))
    axs = axs.flatten()
    for index, col in enumerate(data.columns):
        sns.boxplot(y=col, data=data, ax=axs[index])
    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    st.subheader("Model Training & Evaluation")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    }

    selected_model = st.selectbox("Select a model", list(models.keys()))

    X = data.drop("Potability", axis=1)
    y = data["Potability"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    train_X, test_X, train_y, test_y = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = models[selected_model]
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)

    acc = accuracy_score(test_y, y_pred)
    prec = precision_score(test_y, y_pred, zero_division=0)
    rec = recall_score(test_y, y_pred, zero_division=0)
    f1 = f1_score(test_y, y_pred, zero_division=0)

    try:
        y_proba = model.predict_proba(test_X)[:, 1]
        auc = roc_auc_score(test_y, y_proba)
    except Exception:
        auc = None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.3f}")
    col2.metric("Precision", f"{prec:.3f}")
    col3.metric("Recall", f"{rec:.3f}")
    col4.metric("F1 Score", f"{f1:.3f}")
    if auc is not None:
        st.metric("ROC-AUC", f"{auc:.3f}")

    st.subheader("Classification Report")
    st.text(classification_report(test_y, y_pred))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(test_y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Model Comparison")
    results = []
    for name, m in models.items():
        m.fit(train_X, train_y)
        pred = m.predict(test_X)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(test_y, pred),
            "Precision": precision_score(test_y, pred, zero_division=0),
            "Recall": recall_score(test_y, pred, zero_division=0),
            "F1 Score": f1_score(test_y, pred, zero_division=0),
        })
    results_df = pd.DataFrame(results)
    st.dataframe(results_df.sort_values("Accuracy", ascending=False), use_container_width=True)

    fig = px.bar(results_df, x="Model", y="Accuracy", title="Model Accuracy Comparison")
    st.plotly_chart(fig, use_container_width=True)
