{\rtf1\ansi\ansicpg1252\cocoartf2868
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import pandas as pd\
import numpy as np\
import plotly.express as px\
import plotly.graph_objects as go\
from sklearn.model_selection import train_test_split\
from sklearn.preprocessing import StandardScaler\
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score\
from sklearn.neighbors import KNeighborsClassifier\
from sklearn.tree import DecisionTreeClassifier\
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\
\
# --- PAGE CONFIGURATION ---\
st.set_page_config(page_title="Food Truck Analytics Dashboard", layout="wide")\
st.title("\uc0\u55356 \u57172  Healthy Food Truck - Consumer Analytics Dashboard")\
\
# --- DATA LOADING ---\
@st.cache_data\
def load_data():\
    # Adjust filename if necessary\
    df = pd.read_csv("synthetic_student_survey.xlsx - Sheet1.csv")\
    df['Diet_Restriction'] = df['Diet_Restriction'].fillna('None') # Handle missing values\
    return df\
\
df = load_data()\
\
# --- SIDEBAR NAVIGATION ---\
st.sidebar.header("Navigation")\
menu = st.sidebar.radio(\
    "Select Analysis Phase:",\
    ("1. Descriptive Analytics", "2. Diagnostic Analysis (Bias)", "3. Supervised Machine Learning")\
)\
\
# ==========================================\
# 1. DESCRIPTIVE ANALYTICS\
# ==========================================\
if menu == "1. Descriptive Analytics":\
    st.header("1. Descriptive Analytics (Cross-Tabulations)")\
    st.markdown("Analyzing consumer attributes against their **Subscription (Policy) Status**.")\
    \
    col1, col2 = st.columns(2)\
    \
    with col1:\
        st.subheader("Campus vs. Subscription")\
        ct_campus = pd.crosstab(df['Campus'], df['Subscription'])\
        st.dataframe(ct_campus, use_container_width=True)\
        fig_campus = px.bar(ct_campus, barmode='group', color_discrete_sequence=['#EF553B', '#00CC96'])\
        st.plotly_chart(fig_campus, use_container_width=True)\
\
    with col2:\
        st.subheader("Living Arrangement vs. Subscription")\
        ct_living = pd.crosstab(df['Living_Arrangement'], df['Subscription'])\
        st.dataframe(ct_living, use_container_width=True)\
        fig_living = px.bar(ct_living, barmode='group', color_discrete_sequence=['#EF553B', '#00CC96'])\
        st.plotly_chart(fig_living, use_container_width=True)\
\
# ==========================================\
# 2. DIAGNOSTIC ANALYSIS\
# ==========================================\
elif menu == "2. Diagnostic Analysis (Bias)":\
    st.header("2. Diagnostic Analysis: Identifying Consumer Bias")\
    st.markdown("Proving behavioral and financial biases regarding who subscribes to the food truck.")\
    \
    col1, col2 = st.columns(2)\
    \
    with col1:\
        st.subheader("Income Bias on Subscriptions")\
        fig_income = px.box(df, x="Subscription", y="Food_Income_AED", color="Subscription",\
                            title="Food Income Distribution by Subscription Status")\
        st.plotly_chart(fig_income, use_container_width=True)\
        \
    with col2:\
        st.subheader("Age Bias on Subscriptions")\
        fig_age = px.violin(df, x="Subscription", y="Age", color="Subscription", box=True,\
                            title="Age Distribution by Subscription Status")\
        st.plotly_chart(fig_age, use_container_width=True)\
\
# ==========================================\
# 3. SUPERVISED MACHINE LEARNING\
# ==========================================\
elif menu == "3. Supervised Machine Learning":\
    st.header("3. Supervised Learning: Subscription Prediction")\
    st.markdown("Evaluating KNN, Decision Tree, Random Forest, and Gradient Boosting.")\
    \
    # --- FEATURE ENGINEERING ---\
    st.subheader("Feature Engineering & Preprocessing")\
    st.text("1. Target Variable: 'Subscription' mapped to Binary (0 = No, 1 = Yes)")\
    st.text("2. Categorical Variables: One-Hot Encoded")\
    st.text("3. Numerical Variables: Standard Scaled")\
    \
    # Target\
    y = df['Subscription'].apply(lambda x: 1 if x == 'Yes' else 0)\
    # Drop complex text lists and target\
    X = df.drop(columns=['Subscription', 'Combo_Items'])\
    \
    # One Hot Encode\
    categorical_cols = X.select_dtypes(include=['object']).columns\
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)\
    \
    # Train/Test Split\
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)\
    \
    # Scale\
    scaler = StandardScaler()\
    X_train_scaled = scaler.fit_transform(X_train)\
    X_test_scaled = scaler.transform(X_test)\
    \
    # --- MODEL TRAINING ---\
    models = \{\
        "KNN": KNeighborsClassifier(n_neighbors=5),\
        "Decision Tree": DecisionTreeClassifier(random_state=42),\
        "Random Forest": RandomForestClassifier(random_state=42),\
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)\
    \}\
    \
    results = \{\}\
    roc_data = \{\}\
    cm_data = \{\}\
    \
    for name, model in models.items():\
        model.fit(X_train_scaled, y_train)\
        y_pred = model.predict(X_test_scaled)\
        y_prob = model.predict_proba(X_test_scaled)[:, 1]\
        \
        # Metrics\
        results[name] = \{\
            "Accuracy": accuracy_score(y_test, y_pred),\
            "Precision": precision_score(y_test, y_pred, zero_division=0),\
            "Recall": recall_score(y_test, y_pred, zero_division=0),\
            "F1-Score": f1_score(y_test, y_pred, zero_division=0)\
        \}\
        \
        # ROC Data\
        fpr, tpr, _ = roc_curve(y_test, y_prob)\
        roc_data[name] = (fpr, tpr, auc(fpr, tpr))\
        \
        # Confusion Matrix\
        cm_data[name] = confusion_matrix(y_test, y_pred)\
        \
    # Display Metrics Table\
    results_df = pd.DataFrame(results).T\
    st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'))\
    \
    # Plot Metrics Bar Chart\
    fig_metrics = px.bar(results_df, barmode='group', title="Model Performance Metrics Comparison")\
    st.plotly_chart(fig_metrics, use_container_width=True)\
    \
    # --- ROC CURVE ---\
    st.subheader("ROC Curves (Model Stability Check)")\
    fig_roc = go.Figure()\
    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)\
    \
    for name, (fpr, tpr, roc_auc) in roc_data.items():\
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"\{name\} (AUC = \{roc_auc:.2f\})"))\
        \
    fig_roc.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',\
                          title='Receiver Operating Characteristic (ROC)')\
    st.plotly_chart(fig_roc, use_container_width=True)\
    \
    # --- CONFUSION MATRICES ---\
    st.subheader("Confusion Matrices")\
    cols = st.columns(2)\
    \
    for idx, (name, cm) in enumerate(cm_data.items()):\
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', \
                           title=f"\{name\} Confusion Matrix",\
                           labels=dict(x="Predicted", y="Actual"))\
        cols[idx % 2].plotly_chart(fig_cm, use_container_width=True)}
