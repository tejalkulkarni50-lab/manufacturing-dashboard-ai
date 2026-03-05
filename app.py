import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Manufacturing Dashboard", layout="wide")

st.title("AI Based Manufacturing Efficiency Dashboard")

uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # DATA PREVIEW
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Summary")
    st.write(df.describe())

    # KPI SECTION
    st.subheader("Key Performance Indicators")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Total Machines", df["Machine_ID"].nunique())

    col2.metric("Average Temperature", round(df["Temperature_C"].mean(),2))

    col3.metric("Average Vibration", round(df["Vibration_Hz"].mean(),2))

    col4.metric("Average Error Rate", round(df["Error_Rate_%"].mean(),2))

    # EFFICIENCY DISTRIBUTION
    st.subheader("Efficiency Status Distribution")

    fig = px.histogram(df,x="Efficiency_Stattus",color="Efficiency_Stattus")
    st.plotly_chart(fig,use_container_width=True)

    # PRODUCTION TREND
    st.subheader("Production Speed Trend")

    fig = px.line(df,y="Production_Speed_units_per_hr")
    st.plotly_chart(fig,use_container_width=True)

    # TEMPERATURE DISTRIBUTION
    st.subheader("Temperature Distribution")

    fig = px.histogram(df,x="Temperature_C")
    st.plotly_chart(fig,use_container_width=True)

    # VIBRATION DISTRIBUTION
    st.subheader("Vibration Distribution")

    fig = px.histogram(df,x="Vibration_Hz")
    st.plotly_chart(fig,use_container_width=True)

    # MACHINE WISE PRODUCTION
    st.subheader("Machine Wise Production")

    machine_prod = df.groupby("Machine_ID")["Production_Speed_units_per_hr"].mean().reset_index()

    fig = px.bar(machine_prod,x="Machine_ID",y="Production_Speed_units_per_hr")
    st.plotly_chart(fig,use_container_width=True)

    # CORRELATION HEATMAP
    st.subheader("Feature Correlation Heatmap")

    numeric_df = df.select_dtypes(include=["float64","int64"])

    fig,ax = plt.subplots(figsize=(10,5))
    sns.heatmap(numeric_df.corr(),annot=True,cmap="coolwarm",ax=ax)

    st.pyplot(fig)

    # AI MODEL
    st.subheader("AI Efficiency Prediction Model")

    df = df.dropna()

    X = df[[
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Quality_Control_Defect_Rate_%",
    "Production_Speed_units_per_hr",
    "Predictive_Maintenance_Score",
    "Error_Rate_%"
    ]]

    y = df["Efficiency_Stattus"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    model = RandomForestClassifier()

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test,pred)

    st.success(f"Model Accuracy : {round(acc*100,2)} %")

    # FEATURE IMPORTANCE

    importance = pd.DataFrame({
    "Feature":X.columns,
    "Importance":model.feature_importances_
    }).sort_values(by="Importance",ascending=False)

    st.subheader("Feature Importance")

    fig = px.bar(importance,x="Importance",y="Feature",orientation="h")

    st.plotly_chart(fig,use_container_width=True)

else:

    st.info("Upload your dataset to generate dashboard")
