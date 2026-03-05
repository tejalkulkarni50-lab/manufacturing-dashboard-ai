import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Manufacturing Dashboard", layout="wide")

# ------------------ STYLE ------------------

st.markdown("""
<style>

body {
background-color:#0f172a;
}

.main-title {
font-size:40px;
font-weight:bold;
color:#38bdf8;
text-align:center;
}

.kpi-card {
background: linear-gradient(135deg,#1e293b,#0ea5e9);
padding:20px;
border-radius:15px;
text-align:center;
color:white;
font-size:18px;
box-shadow:0px 4px 12px rgba(0,0,0,0.3);
}

.section-title {
color:#22c55e;
font-size:24px;
font-weight:bold;
margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------

st.markdown('<p class="main-title">🏭 AI Smart Manufacturing Efficiency Dashboard</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # CLEAN COLUMN NAMES
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(" ", "_")

    # ---------------- DATA PREVIEW ----------------

    st.markdown('<p class="section-title">Dataset Preview</p>', unsafe_allow_html=True)
    st.dataframe(df.head())

    st.markdown('<p class="section-title">Dataset Summary</p>', unsafe_allow_html=True)
    st.write(df.describe())

    # ---------------- KPI SECTION ----------------

    st.markdown('<p class="section-title">Key Performance Indicators</p>', unsafe_allow_html=True)

    col1,col2,col3,col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
        Total Machines
        <h2>{df["Machine_ID"].nunique()}</h2>
        </div>
        """,unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
        Avg Temperature
        <h2>{round(df["Temperature_C"].mean(),2)}</h2>
        </div>
        """,unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
        Avg Vibration
        <h2>{round(df["Vibration_Hz"].mean(),2)}</h2>
        </div>
        """,unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
        Avg Error Rate
        <h2>{round(df["Error_Rate_%"].mean(),2)}</h2>
        </div>
        """,unsafe_allow_html=True)

    # ---------------- EFFICIENCY DISTRIBUTION ----------------

    eff_col = [c for c in df.columns if "eff" in c.lower()]

    if eff_col:

        st.markdown('<p class="section-title">Efficiency Distribution</p>', unsafe_allow_html=True)

        fig = px.histogram(
            df,
            x=eff_col[0],
            color=eff_col[0],
            color_discrete_sequence=["#06b6d4","#22c55e","#f59e0b"]
        )

        st.plotly_chart(fig,use_container_width=True)

    # ---------------- PRODUCTION TREND ----------------

    prod_col = [c for c in df.columns if "production" in c.lower()]

    if prod_col:

        st.markdown('<p class="section-title">Production Trend</p>', unsafe_allow_html=True)

        fig = px.line(
            df,
            y=prod_col[0],
            color_discrete_sequence=["#38bdf8"]
        )

        st.plotly_chart(fig,use_container_width=True)

    # ---------------- TEMPERATURE ----------------

    if "Temperature_C" in df.columns:

        st.markdown('<p class="section-title">Temperature Distribution</p>', unsafe_allow_html=True)

        fig = px.histogram(
            df,
            x="Temperature_C",
            color_discrete_sequence=["#f97316"]
        )

        st.plotly_chart(fig,use_container_width=True)

    # ---------------- VIBRATION ----------------

    if "Vibration_Hz" in df.columns:

        st.markdown('<p class="section-title">Vibration Distribution</p>', unsafe_allow_html=True)

        fig = px.histogram(
            df,
            x="Vibration_Hz",
            color_discrete_sequence=["#22c55e"]
        )

        st.plotly_chart(fig,use_container_width=True)

    # ---------------- MACHINE PRODUCTION ----------------

    if "Machine_ID" in df.columns and prod_col:

        st.markdown('<p class="section-title">Machine Wise Production</p>', unsafe_allow_html=True)

        machine_prod = df.groupby("Machine_ID")[prod_col[0]].mean().reset_index()

        fig = px.bar(
            machine_prod,
            x="Machine_ID",
            y=prod_col[0],
            color_discrete_sequence=["#6366f1"]
        )

        st.plotly_chart(fig,use_container_width=True)

    # ---------------- HEATMAP ----------------

    st.markdown('<p class="section-title">Feature Correlation Heatmap</p>', unsafe_allow_html=True)

    numeric_df = df.select_dtypes(include=["float64","int64"])

    if len(numeric_df.columns) > 1:

        fig,ax = plt.subplots(figsize=(10,5))

        sns.heatmap(numeric_df.corr(),annot=True,cmap="coolwarm",ax=ax)

        st.pyplot(fig)

    # ---------------- AI MODEL ----------------

    if eff_col:

        st.markdown('<p class="section-title">AI Efficiency Prediction Model</p>', unsafe_allow_html=True)

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

        st.markdown('<p class="section-title">Feature Importance</p>', unsafe_allow_html=True)

        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            color_discrete_sequence=["#0ea5e9"]
        )

        st.plotly_chart(fig,use_container_width=True)

else:

    st.info("Upload dataset to generate dashboard")
