import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
st.set_page_config(page_title="AI Manufacturing Efficiency Dashboard", layout="wide")

st.title("🏭 AI-Based Manufacturing Efficiency Dashboard")

uploaded_file = st.file_uploader("Upload Manufacturing Dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Summary")
    st.write(df.describe())

    # ---------------- KPI SECTION ----------------
    st.subheader("📊 Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    if "Production_Count" in df.columns:
        col1.metric("Total Production", int(df["Production_Count"].sum()))

    if "Error_Rate" in df.columns:
        col2.metric("Average Error Rate", round(df["Error_Rate"].mean(),2))

    if "Sensor_Value" in df.columns:
        col3.metric("Average Sensor Value", round(df["Sensor_Value"].mean(),2))

    if "Efficiency_Class" in df.columns:
        high_eff = (df["Efficiency_Class"] == "High").sum()
        col4.metric("High Efficiency Machines", high_eff)

    # ---------------- GRAPHS ----------------

    st.subheader("📈 Efficiency Class Distribution")

    if "Efficiency_Class" in df.columns:
        fig, ax = plt.subplots()
        sns.countplot(x="Efficiency_Class", data=df, palette="viridis", ax=ax)
        st.pyplot(fig)

    st.subheader("📈 Production Trend")

    if "Production_Count" in df.columns:
        fig, ax = plt.subplots()
        df["Production_Count"].plot(ax=ax)
        ax.set_ylabel("Production")
        st.pyplot(fig)

    st.subheader("📊 Sensor Value Distribution")

    if "Sensor_Value" in df.columns:
        fig, ax = plt.subplots()
        sns.histplot(df["Sensor_Value"], kde=True, ax=ax)
        st.pyplot(fig)

    # ---------------- MACHINE ANALYSIS ----------------

    if "Machine_ID" in df.columns and "Production_Count" in df.columns:
        st.subheader("🏭 Machine Wise Production")

        machine_prod = df.groupby("Machine_ID")["Production_Count"].sum()

        fig, ax = plt.subplots()
        machine_prod.plot(kind="bar", ax=ax)
        st.pyplot(fig)

    # ---------------- HEATMAP ----------------

    st.subheader("🔥 Feature Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # ---------------- MACHINE LEARNING ----------------

    st.subheader("🤖 AI Efficiency Prediction Model")

    if "Efficiency_Class" in df.columns:

        df = df.dropna()

        X = df.select_dtypes(include=['int64','float64'])
        y = df["Efficiency_Class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,y,test_size=0.2,random_state=42
        )

        model = RandomForestClassifier(n_estimators=200)

        model.fit(X_train,y_train)

        predictions = model.predict(X_test)

        acc = accuracy_score(y_test,predictions)

        st.success(f"Model Accuracy: {round(acc*100,2)} %")

        st.subheader("Classification Report")
        st.text(classification_report(y_test,predictions))

        # -------- Feature Importance --------

        st.subheader("📊 Feature Importance")

        importance = pd.Series(model.feature_importances_, index=X.columns)

        fig, ax = plt.subplots()

        importance.sort_values().plot(kind='barh', ax=ax)

        st.pyplot(fig)

else:

    st.info("Upload your dataset to generate AI Manufacturing Dashboard")
