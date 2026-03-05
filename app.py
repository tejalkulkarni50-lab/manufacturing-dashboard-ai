import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Manufacturing Dashboard", layout="wide")

st.title("🏭 AI-Based Manufacturing Efficiency Dashboard")

uploaded_file = st.file_uploader("Upload Manufacturing Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    df.columns = df.columns.str.strip()

    # ---------------- Dataset Preview ----------------

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Summary")
    st.write(df.describe())

    # ---------------- KPI Section ----------------

    st.subheader("Key Performance Indicators")

    col1,col2,col3,col4 = st.columns(4)

    if "Production_Count" in df.columns:
        col1.metric("Total Production", int(df["Production_Count"].sum()))

    if "Error_Rate" in df.columns:
        col2.metric("Average Error Rate", round(df["Error_Rate"].mean(),2))

    if "Sensor_Value" in df.columns:
        col3.metric("Average Sensor Value", round(df["Sensor_Value"].mean(),2))

    if "Efficiency_Class" in df.columns:
        high=(df["Efficiency_Class"]=="High").sum()
        col4.metric("High Efficiency Machines", high)

    # ---------------- Efficiency Distribution ----------------

    st.subheader("Efficiency Class Distribution")

    if "Efficiency_Class" in df.columns:

        fig = px.histogram(df,x="Efficiency_Class",color="Efficiency_Class")

        st.plotly_chart(fig,use_container_width=True)

    # ---------------- Production Trend ----------------

    st.subheader("Production Trend")

    if "Production_Count" in df.columns:

        fig = px.line(df,y="Production_Count")

        st.plotly_chart(fig,use_container_width=True)

    # ---------------- Sensor Distribution ----------------

    st.subheader("Sensor Value Distribution")

    if "Sensor_Value" in df.columns:

        fig = px.histogram(df,x="Sensor_Value")

        st.plotly_chart(fig,use_container_width=True)

    # ---------------- Machine Production ----------------

    st.subheader("Machine Wise Production")

    if "Machine_ID" in df.columns and "Production_Count" in df.columns:

        machine_prod = df.groupby("Machine_ID")["Production_Count"].sum().reset_index()

        fig = px.bar(machine_prod,x="Machine_ID",y="Production_Count")

        st.plotly_chart(fig,use_container_width=True)

    # ---------------- Correlation Heatmap ----------------

    st.subheader("Feature Correlation Heatmap")

    numeric_df = df.select_dtypes(include=["float64","int64"])

    fig,ax = plt.subplots(figsize=(10,5))

    sns.heatmap(numeric_df.corr(),annot=True,cmap="coolwarm",ax=ax)

    st.pyplot(fig)

    # ---------------- AI Model ----------------

    st.subheader("AI Efficiency Prediction Model")

    if "Efficiency_Class" in df.columns:

        df=df.dropna()

        X=df.select_dtypes(include=["float64","int64"])

        y=df["Efficiency_Class"]

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

        model=RandomForestClassifier()

        model.fit(X_train,y_train)

        pred=model.predict(X_test)

        acc=accuracy_score(y_test,pred)

        st.success(f"Model Accuracy : {round(acc*100,2)} %")

        # Feature Importance

        importance=pd.DataFrame({
        "Feature":X.columns,
        "Importance":model.feature_importances_
        })

        importance=importance.sort_values(by="Importance",ascending=False)

        st.subheader("Feature Importance")

        fig = px.bar(importance,x="Importance",y="Feature",orientation="h")

        st.plotly_chart(fig,use_container_width=True)

else:

    st.info("Upload dataset to generate AI dashboard")
