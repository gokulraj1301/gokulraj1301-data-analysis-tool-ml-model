import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from fpdf import FPDF
from io import BytesIO

st.set_page_config(page_title="Data Insights & ML Tool", layout="wide")
st.title("📊 Data Analysis & ML Insight Generator")

# PDF Report Generator (without emojis)
def generate_pdf_report(df, summary_text):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Dataset Analysis Report", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Rows: {df.shape[0]}, Columns: {df.shape[1]}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Column Types:", ln=True)
    pdf.set_font("Arial", "", 10)
    for col, dtype in df.dtypes.items():
        pdf.cell(0, 8, f"{col}: {dtype}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Summary Insights:", ln=True)
    pdf.set_font("Arial", "", 11)
    for line in summary_text.split("\n"):
        pdf.multi_cell(0, 8, line)

    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    st.write("### 🧾 Preview of Uploaded Dataset")
    st.dataframe(df.head())

    st.markdown("---")

    st.write("## 📋 Dataset Summary")
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Shape")
        st.write(f"Rows: {df.shape[0]}  \nColumns: {df.shape[1]}")

        st.write("### Column Types")
        st.write(df.dtypes)

    with col2:
        st.write("### Descriptive Stats (Numerical)")
        st.dataframe(df.describe().T)

    st.write("### Missing Values")
    st.dataframe(df.isnull().sum()[df.isnull().sum() > 0])

    st.markdown("---")

    st.header("📊 Data Visualizations")

    numeric_df = df.select_dtypes(include=['number'])
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if numeric_df.shape[1] >= 2:
        st.subheader("Correlation Heatmap (numeric only)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns for correlation heatmap.")

    if cat_cols:
        st.subheader("Categorical Distributions")
        for col in cat_cols[:2]:
            st.markdown(f"**{col}**")
            fig, ax = plt.subplots(figsize=(6, 3))
            df[col].value_counts().head(10).plot(kind='bar', ax=ax, color='skyblue')
            ax.set_ylabel("Count")
            ax.set_xlabel(col)
            st.pyplot(fig)

    st.markdown("---")

    st.subheader("💡 Automated Insights Summary")
    num_summary = df.describe().T
    top_numeric = num_summary.sort_values(by="std", ascending=False).head(2).index.tolist()
    cat_summary = [col for col in cat_cols if df[col].nunique() < 10]

    insights_text = ""

    if top_numeric:
        insights_text += f"- {top_numeric[0]} and {top_numeric[1]} show the highest variability.\n"
    if cat_summary:
        insights_text += f"- {cat_summary[0]} has {df[cat_summary[0]].nunique()} unique values with '{df[cat_summary[0]].mode()[0]}' most common.\n"

    if "income" in df.columns:
        insights_text += "- Income-related attributes, useful for segmentation or salary analysis.\n"
    elif "sales" in df.columns:
        insights_text += "- Sales data, potentially for forecasting or inventory.\n"
    else:
        insights_text += "- General tabular dataset.\n"

    st.text(insights_text)

    # PDF Download Button
    pdf_bytes = generate_pdf_report(df, insights_text)
    st.download_button("📥 Download Insights as PDF", data=pdf_bytes.getvalue(),
                       file_name="data_insights_report.pdf", mime="application/pdf")

    st.markdown("---")

    st.subheader("🎯 Choose your target column for ML modeling (optional)")
    target = st.selectbox("Select target column (or skip for just analysis):", [""] + list(df.columns))

    if target != "":
        st.subheader(f"🧠 ML Model: Predicting `{target}`")
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
