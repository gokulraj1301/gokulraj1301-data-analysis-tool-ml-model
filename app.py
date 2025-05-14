import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="📊 Data Insight & ML Tool", layout="wide")

# File uploader
st.title("📊 Smart Data Analyzer & ML Assistant")
uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 📋 Preview of Uploaded Data")
    st.dataframe(df.head())

    st.write("### 🧮 Dataset Summary Statistics")
    st.dataframe(df.describe(include='all'))

    st.write("### 🧯 Missing Values Check")
    st.dataframe(df.isnull().sum())

    st.markdown("---")
    st.subheader("📊 Compact Visual Insights")

    # Correlation Heatmap
    st.write("#### 🔗 Correlation Heatmap (Numeric Features Only)")
    numeric_df = df.select_dtypes(include=['number'])

    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(6, 4))  # Smaller figure
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns for a heatmap.")

    # Top Categorical Distributions
    st.write("#### 🔠 Top Categorical Feature Distributions")
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    if cat_cols:
        for col in cat_cols[:2]:  # Only top 2 for compactness
            st.write(f"📌 {col} value counts")
            fig, ax = plt.subplots(figsize=(6, 3))
            df[col].value_counts().head(10).plot(kind='bar', ax=ax)
            ax.set_ylabel("Count")
            st.pyplot(fig)
    else:
        st.info("No categorical columns available to visualize.")

    st.markdown("---")
    st.subheader("🧠 Automatic Insight Summary")

    # Basic data storytelling
    st.write("📌 **Data Insights Summary**")

    total_rows = df.shape[0]
    total_cols = df.shape[1]
    num_missing = df.isnull().sum().sum()
    num_cols = len(numeric_df.columns)
    cat_cols_count = len(cat_cols)

    st.markdown(f"""
    - 🔢 The dataset contains **{total_rows} rows** and **{total_cols} columns**.
    - 🧩 **{num_cols} numeric columns** and **{cat_cols_count} categorical columns** detected.
    - 🕳️ Total **missing values** in the dataset: `{num_missing}`
    - 📈 Top correlated features hint at possible relationships that may impact predictions.
    - 📊 Category distribution plots reveal how values are concentrated (imbalanced or dominant).
    """)

    # ML Modeling Section
    st.markdown("---")
    st.subheader("🤖 Optional: ML Model Training")

    target = st.text_input("🎯 Enter your target column (leave blank to skip modeling):")

    if target:
        if target in df.columns:
            st.write(f"### 🔍 Modeling to Predict `{target}`")

            try:
                X = df.drop(columns=[target])
                y = df[target]

                X = pd.get_dummies(X, drop_first=True)

                if not pd.api.types.is_numeric_dtype(y):
                    y = pd.factorize(y)[0]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = RandomForestRegressor()
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                st.success(f"✅ Model Trained Successfully | RMSE: {rmse:.2f}")

                importances = model.feature_importances_
                feat_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
                feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

                st.write("### 🔍 Feature Importances")
                st.dataframe(feat_imp_df.head(10))

                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x='Importance', y='Feature', data=feat_imp_df.head(10), ax=ax)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"⚠️ Error during modeling: {e}")
        else:
            st.error("❌ Target column not found in the dataset.")
else:
    st.info("⬆️ Please upload a CSV file to begin analysis.")
