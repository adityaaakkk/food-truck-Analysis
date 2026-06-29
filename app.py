import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Food Truck Analytics Dashboard", layout="wide")
st.title("🍔 Healthy Food Truck - Consumer Analytics Dashboard")

# --- DATA LOADING (BULLETPROOF) ---
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_student_survey.xlsx - Sheet1.csv")
    # Clean hidden spaces from column names completely
    df.columns = df.columns.str.strip() 
    df = df.dropna(how='all') 
    
    # Dynamically find the target column name even if it has minor typos or hidden characters
    target_col = None
    for col in df.columns:
        if 'subscription' in col.lower():
            target_col = col
            break
            
    if target_col and target_col != 'Subscription':
        df = df.rename(columns={target_col: 'Subscription'})
        
    if 'Diet_Restriction' in df.columns:
        df['Diet_Restriction'] = df['Diet_Restriction'].fillna('None') 
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data. Error details: {e}")
    st.stop()

# Ensure target column exists globally or throw descriptive fallback
if 'Subscription' not in df.columns:
    st.error(f"Could not find the target column 'Subscription' in your dataset. Available columns are: {list(df.columns)}")
    st.stop()

# --- SIDEBAR NAVIGATION ---
st.sidebar.header("Navigation")
menu = st.sidebar.radio(
    "Select Analysis Phase:",
    ("1. Descriptive Analytics", "2. Diagnostic Analysis (Bias)", "3. Supervised Machine Learning")
)

# ==========================================
# 1. DESCRIPTIVE ANALYTICS
# ==========================================
if menu == "1. Descriptive Analytics":
    st.header("1. Descriptive Analytics (Cross-Tabulations)")
    st.markdown("Analyzing consumer attributes against their **Subscription Status**.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Campus vs. Subscription")
        if 'Campus' in df.columns:
            ct_campus = pd.crosstab(df['Campus'], df['Subscription'])
            st.dataframe(ct_campus, use_container_width=True)
            fig_campus = px.bar(ct_campus, barmode='group', color_discrete_sequence=['#EF553B', '#00CC96'])
            st.plotly_chart(fig_campus, use_container_width=True)

    with col2:
        st.subheader("Living Arrangement vs. Subscription")
        if 'Living_Arrangement' in df.columns:
            ct_living = pd.crosstab(df['Living_Arrangement'], df['Subscription'])
            st.dataframe(ct_living, use_container_width=True)
            fig_living = px.bar(ct_living, barmode='group', color_discrete_sequence=['#EF553B', '#00CC96'])
            st.plotly_chart(fig_living, use_container_width=True)

# ==========================================
# 2. DIAGNOSTIC ANALYSIS
# ==========================================
elif menu == "2. Diagnostic Analysis (Bias)":
    st.header("2. Diagnostic Analysis: Identifying Consumer Bias")
    st.markdown("Proving behavioral and financial biases regarding who subscribes to the food truck.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Income Bias on Subscriptions")
        if 'Food_Income_AED' in df.columns:
            fig_income = px.box(df, x="Subscription", y="Food_Income_AED", color="Subscription",
                                title="Food Income Distribution by Subscription Status")
            st.plotly_chart(fig_income, use_container_width=True)
        
    with col2:
        st.subheader("Age Bias on Subscriptions")
        if 'Age' in df.columns:
            fig_age = px.violin(df, x="Subscription", y="Age", color="Subscription", box=True,
                                title="Age Distribution by Subscription Status")
            st.plotly_chart(fig_age, use_container_width=True)

# ==========================================
# 3. SUPERVISED MACHINE LEARNING
# ==========================================
elif menu == "3. Supervised Machine Learning":
    st.header("3. Supervised Learning: Subscription Prediction")
    st.markdown("Evaluating KNN, Decision Tree, Random Forest, and Gradient Boosting.")
    
    df_ml = df.dropna(subset=['Subscription']).copy()
    y = df_ml['Subscription'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
    
    cols_to_drop = ['Subscription', 'Combo_Items']
    cols_to_drop = [c for c in cols_to_drop if c in df_ml.columns]
    X = df_ml.drop(columns=cols_to_drop)
    
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna('Unknown')
        else:
            X[col] = X[col].fillna(0)
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    roc_data = {}
    cm_data = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0)
        }
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, auc(fpr, tpr))
        cm_data[name] = confusion_matrix(y_test, y_pred)
        
    results_df = pd.DataFrame(results).T
    st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'))
    
    fig_metrics = px.bar(results_df, barmode='group', title="Model Performance Metrics Comparison")
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    st.subheader("ROC Curves (Model Stability Check)")
    fig_roc = go.Figure()
    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    
    for name, (fpr, tpr, roc_auc) in roc_data.items():
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC = {roc_auc:.2f})"))
        
    fig_roc.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', title='Receiver Operating Characteristic (ROC)')
    st.plotly_chart(fig_roc, use_container_width=True)
    
    st.subheader("Confusion Matrices")
    cols = st.columns(2)
    for idx, (name, cm) in enumerate(cm_data.items()):
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title=f"{name} Matrix", labels=dict(x="Predicted", y="Actual"))
        cols[idx % 2].plotly_chart(fig_cm, use_container_width=True)
