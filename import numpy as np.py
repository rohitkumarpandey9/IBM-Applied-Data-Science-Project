import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load historical stock data (example data)
# Replace this with your actual stock data
# The dataset should have columns 'Date' and 'Close'
# Example:
# data = pd.read_csv('stock_data.csv')
# For simplicity, we'll create a dummy dataset.
data = pd.DataFrame({'Date': pd.date_range(start='1/1/2010', periods=100, freq='D'),
                     'Close': np.random.randint(50, 300, 100)})

# Use only the 'Close' prices for prediction
data = data[['Close']]

# Create a new column for predicting 'n' days into the future
n = 30  # Number of days to predict into the future
data['Prediction'] = data['Close'].shift(-n)

# Create feature set (X) and target set (y)
X = np.array(data.drop(['Prediction'], 1))
X = X[:-n]  # Remove the last 'n' rows as we don't have predictions for them
y = np.array(data['Prediction'])[:-n]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
predictions = model.predict(X_test)

# Visualize the predictions
plt.figure(figsize=(12, 6))
plt.plot(predictions, label='Predicted')
plt.plot(y_test, label='Actual')
plt.legend()
plt.show()
