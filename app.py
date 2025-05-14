import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="📊 Data Insights & ML Tool", layout="wide")

st.title("📊 Data Analysis & ML Insight Generator")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    st.write("### 🧾 Preview of Uploaded Dataset")
    st.dataframe(df.head())

    st.markdown("---")

    # Dataset Summary
    st.write("## 📋 Dataset Summary")
    col1, col2 = st.columns(2)

    with col1:
        st.write("### 📌 Shape")
        st.write(f"Rows: {df.shape[0]}  \nColumns: {df.shape[1]}")

        st.write("### 🧱 Column Types")
        st.write(df.dtypes)

    with col2:
        st.write("### 🧮 Descriptive Stats (Numerical)")
        st.dataframe(df.describe().T)

    st.write("### 🧼 Missing Values")
    st.dataframe(df.isnull().sum()[df.isnull().sum() > 0])

    st.markdown("---")

    # Data Visualizations
    st.header("📊 Data Visualizations")

    numeric_df = df.select_dtypes(include=['number'])
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if numeric_df.shape[1] >= 2:
        st.subheader("📌 Correlation Heatmap (numeric only)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns for correlation heatmap.")

    if cat_cols:
        st.subheader("📌 Categorical Distributions")
        for col in cat_cols[:2]:  # Limit to 2 top categorical features
            st.markdown(f"**{col}**")
            fig, ax = plt.subplots(figsize=(6, 3))
            df[col].value_counts().head(10).plot(kind='bar', ax=ax, color='skyblue')
            ax.set_ylabel("Count")
            ax.set_xlabel(col)
            st.pyplot(fig)

    st.markdown("---")

    # Simple Automated Insights (Text Summary)
    st.subheader("💡 Automated Insights Summary")
    num_summary = df.describe().T
    top_numeric = num_summary.sort_values(by="std", ascending=False).head(2).index.tolist()
    cat_summary = [col for col in cat_cols if df[col].nunique() < 10]

    if top_numeric:
        st.write(f"- **{top_numeric[0]}** and **{top_numeric[1]}** show the highest variability across rows.")
    if cat_summary:
        st.write(f"- **{cat_summary[0]}** has {df[cat_summary[0]].nunique()} unique values with '{df[cat_summary[0]].mode()[0]}' being the most common.")

    st.write("This dataset appears to focus on:")
    if "income" in df.columns:
        st.markdown("- 💰 **Income-related attributes**, useful for customer segmentation or salary analysis.")
    elif "sales" in df.columns:
        st.markdown("- 🛍️ **Sales data**, potentially for forecasting or inventory decisions.")
    else:
        st.write("- Generic tabular data that can be further analyzed by selecting a target column below.")

    st.markdown("---")

    # Target Column Selection (Dropdown for safety)
    st.subheader("🎯 Choose your target column for ML modeling (optional)")
    target = st.selectbox("Select target column (or skip for just analysis):", [""] + list(df.columns))

    if target != "":
        st.subheader(f"🧠 ML Model: Predicting `{target}`")
        try:
            X = df.drop(columns=[target])
            y = df[target]

            X = pd.get_dummies(X, drop_first=True)

            # Encode target if needed
            if not pd.api.types.is_numeric_dtype(y):
                y = pd.factorize(y)[0]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            st.success(f"✅ Model Trained. RMSE: {rmse:.2f}")

            importances = model.feature_importances_
            feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
            feat_imp = feat_imp.sort_values(by='Importance', ascending=False)

            st.write("### 🔍 Top Feature Importances")
            st.dataframe(feat_imp.head(10))

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x='Importance', y='Feature', data=feat_imp.head(10), ax=ax, palette='viridis')
            st.pyplot(fig)

        except Exception as e:
            st.error(f"⚠️ Error during modeling: {e}")
else:
    st.info("⬆️ Upload a CSV file to begin.")
