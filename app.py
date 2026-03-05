import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Manufacturing Dashboard", layout="wide")

st.title("🏭 AI Based Manufacturing Efficiency Dashboard")

uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # CLEAN COLUMN NAMES
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(" ", "_")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Summary")
    st.write(df.describe())

    # ---------------- KPI SECTION ----------------

    st.subheader("Key Performance Indicators")

    col1,col2,col3,col4 = st.columns(4)

    if "Machine_ID" in df.columns:
        col1.metric("Total Machines", df["Machine_ID"].nunique())

    if "Temperature_C" in df.columns:
        col2.metric("Avg Temperature", round(df["Temperature_C"].mean(),2))

    if "Vibration_Hz" in df.columns:
        col3.metric("Avg Vibration", round(df["Vibration_Hz"].mean(),2))

    if "Error_Rate_%" in df.columns:
        col4.metric("Avg Error Rate", round(df["Error_Rate_%"].mean(),2))

    # ---------------- EFFICIENCY DISTRIBUTION ----------------

    eff_col = [c for c in df.columns if "eff" in c.lower()]

    if eff_col:

        st.subheader("Efficiency Distribution")

        fig = px.histogram(df,x=eff_col[0],color=eff_col[0])
        st.plotly_chart(fig,use_container_width=True)

    # ---------------- PRODUCTION TREND ----------------

    prod_col = [c for c in df.columns if "production" in c.lower()]

    if prod_col:

        st.subheader("Production Trend")

        fig = px.line(df,y=prod_col[0])
        st.plotly_chart(fig,use_container_width=True)

    # ---------------- TEMPERATURE ----------------

    if "Temperature_C" in df.columns:

        st.subheader("Temperature Distribution")

        fig = px.histogram(df,x="Temperature_C")
        st.plotly_chart(fig,use_container_width=True)

    # ---------------- VIBRATION ----------------

    if "Vibration_Hz" in df.columns:

        st.subheader("Vibration Distribution")

        fig = px.histogram(df,x="Vibration_Hz")
        st.plotly_chart(fig,use_container_width=True)

    # ---------------- MACHINE PRODUCTION ----------------

    if "Machine_ID" in df.columns and prod_col:

        st.subheader("Machine Wise Production")

        machine_prod = df.groupby("Machine_ID")[prod_col[0]].mean().reset_index()

        fig = px.bar(machine_prod,x="Machine_ID",y=prod_col[0])

        st.plotly_chart(fig,use_container_width=True)

    # ---------------- HEATMAP ----------------

    st.subheader("Feature Correlation Heatmap")

    numeric_df = df.select_dtypes(include=["float64","int64"])

    if len(numeric_df.columns) > 1:

        fig,ax = plt.subplots(figsize=(10,5))

        sns.heatmap(numeric_df.corr(),annot=True,cmap="coolwarm",ax=ax)

        st.pyplot(fig)

    # ---------------- AI MODEL ----------------

    if eff_col:

        st.subheader("AI Efficiency Prediction Model")

        df = df.dropna()

        X = numeric_df

        y = df[eff_col[0]]

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        model = RandomForestClassifier()

        model.fit(X_train,y_train)

        pred = model.predict(X_test)

        acc = accuracy_score(y_test,pred)

        st.success(f"Model Accuracy : {round(acc*100,2)} %")

        importance = pd.DataFrame({
            "Feature":X.columns,
            "Importance":model.feature_importances_
        }).sort_values(by="Importance",ascending=False)

        st.subheader("Feature Importance")

        fig = px.bar(importance,x="Importance",y="Feature",orientation="h")

        st.plotly_chart(fig,use_container_width=True)

else:

    st.info("Upload dataset to generate dashboard")
