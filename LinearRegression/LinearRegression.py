import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Load data
df = pd.read_excel("car_sales_data.xlsx")
# Scatter plot
plt.scatter(df.Mileage, df.Sell_Price)
# Feature and target
Feature = df["Mileage"].values.reshape(-1, 1)  # Reshape to 2D array
target = df["Sell_Price"]

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Feature, target, test_size=0.2, random_state=42)

# Linear Regression model
from sklearn.linear_model import LinearRegression
regObj = LinearRegression()
regObj.fit(X_train, y_train)

# Plot regression line
plt.plot(df.Mileage, regObj.predict(Feature), color='red')
plt.show()

