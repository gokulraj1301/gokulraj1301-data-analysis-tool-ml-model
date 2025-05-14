import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Data Analysis & Insights Tool", layout="wide")

# File uploader
st.title("📊 Automated Data Analysis & ML Insights Tool")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 🧾 Uploaded Dataset")
    st.dataframe(df.head())

    st.write("### 📈 Data Summary")
    st.write(df.describe(include='all'))

    st.write("### 🕳️ Missing Values")
    st.write(df.isnull().sum())

    # Data visualization section
    st.write("## 📊 Data Visualizations")

    # Correlation Heatmap for numerical data
    st.write("### 🔥 Correlation Heatmap (numeric features only)")
    numeric_df = df.select_dtypes(include=['number'])

    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("❌ Not enough numeric columns to compute a correlation heatmap.")

    # Bar plot for top categorical columns
    st.write("### 📊 Top Categorical Feature Distributions")
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    if cat_cols:
        for col in cat_cols[:3]:  # show top 3 categories
            st.write(f"#### {col}")
            fig, ax = plt.subplots()
            df[col].value_counts().head(10).plot(kind='bar', ax=ax)
            st.pyplot(fig)
    else:
        st.info("No categorical columns found to visualize.")

    # Target input
    target = st.text_input("🎯 Enter your target column for ML modeling (leave blank to skip):")

    if target:
        if target in df.columns:
            st.write(f"### 🤖 ML Model: Predicting `{target}`")

            try:
                X = df.drop(columns=[target])
                y = df[target]

                # Convert categorical variables
                X = pd.get_dummies(X, drop_first=True)

                # Remove non-numeric targets if regression
                if not pd.api.types.is_numeric_dtype(y):
                    y = pd.factorize(y)[0]

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train Random Forest model
                model = RandomForestRegressor()
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                st.success(f"✅ Model Trained. RMSE: {rmse:.2f}")

                # Feature importances
                importances = model.feature_importances_
                feat_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
                feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

                st.write("### 💡 Feature Importances")
                st.dataframe(feat_imp_df.head(10))

                fig, ax = plt.subplots()
                sns.barplot(x='Importance', y='Feature', data=feat_imp_df.head(10), ax=ax)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"⚠️ Error during modeling: {e}")
        else:
            st.error("❌ Target column not found in the dataset.")
else:
    st.info("⬆️ Upload a CSV file to begin.")
