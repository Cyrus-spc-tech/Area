import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Price'] = data.target  

st.write("First five rows of dataset:")
st.write(df.head())

X = df.drop(columns=['Price'])  
y = df['Price']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"Mean Squared Error: {mse:.4f}")
st.write(f"R² Score: {r2:.4f}")

house_size = st.number_input("Enter the house size in square feet:", min_value=0.0, step=1.0)

if house_size == 0:
    predicted_price = 0
else:
    average_sqft_per_room = 500  
    avg_household_size = df['AveOccup'].mean()
    median_income = df['MedInc'].mean()

    user_input = np.array([
        house_size / average_sqft_per_room,  # AveRooms
        avg_household_size,  # AveOccup
        median_income,  # MedInc
        X.mean().iloc[3],  
        X.mean().iloc[4],
        X.mean().iloc[5],
        X.mean().iloc[6],
        X.mean().iloc[7]
    ]).reshape(1, -1)

    # Scale user input
    user_input_scaled = scaler.transform(user_input)

    # Predict price
    predicted_price = model.predict(user_input_scaled)[0] * 100000  # Convert to actual price range

st.write(f"Estimated House Price: ${predicted_price:.2f}")

# Visualizing actual vs predicted prices
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, alpha=0.5, color='blue')
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
ax.set_xlabel("Actual Prices")
ax.set_ylabel("Predicted Prices")
ax.set_title("Actual vs Predicted House Prices")
st.pyplot(fig)