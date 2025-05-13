import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# File uploader widget
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Show the uploaded data
    st.write("### Uploaded Data", df.head())

    # Basic data analysis
    st.write("### Data Summary")
    st.write(df.describe())  # Basic summary statistics
    st.write("### Missing Values")
    st.write(df.isnull().sum())  # Check for missing values

    # Data visualization
    st.write("### Data Visualizations")
    st.write("#### Correlation Heatmap")
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    st.pyplot()

    # Pairplot (if applicable)
    st.write("#### Pairplot")
    sns.pairplot(df)
    st.pyplot()

    # Asking for target column for ML modeling
    target = st.text_input("Enter your target column (or leave blank to skip ML modeling):")

    if target:
        if target in df.columns:
            st.write(f"### ML Model: Predicting {target}")
            
            # Example model training (Random Forest as placeholder)
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error

            # Preparing features and target
            X = df.drop(columns=[target])
            y = df[target]

            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Initialize and train the model
            model = RandomForestRegressor()
            model.fit(X_train, y_train)

            # Make predictions and evaluate the model
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            st.write(f"Root Mean Squared Error (RMSE): {rmse}")

            # Display feature importances
            feature_importances = model.feature_importances_
            feature_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False)

            st.write("### Feature Importances")
            st.write(feature_df)

        else:
            st.write("The entered target column does not exist in the dataset.")
