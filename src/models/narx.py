#uses the past states to predict the current one

"""
potential problems: cG values very small?


TODO: maybe try better scaling?
    - maybe not fix the lags at 5, find a better one
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from src.data.prepare_data import create_mimo_lagged_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

file_name = "data_generic.csv"
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", file_name))
#print(file_path)
df = pd.read_csv(file_path)


#print(df.head())

features = ["cA", "cB", "cC", "cP", "cE", "cG"]
exogenous_inputs = ["QB", "Tr"]

#X: past five states, y: current state
X, y, feature_names = create_mimo_lagged_data(df, features, exogenous_inputs)

#first 80% used for training, last 20% for testing
train_size = int(len(X) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]


model_name = "narx_random_forest.pkl"
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", model_name))
#check if model exists, if yes, use that, else train a new one and save it

model = None

if os.path.exists(model_path):
    print("Model already trained")
    model = joblib.load(model_path)
else:
    print("No trained model, training new one and saving it")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
for feature, error in zip(features, mse):
    print(f"Test MSE for {feature}: {error:.10f}")

comparison = pd.DataFrame({
    'Actual': [f"{val:.10f}" for val in y_test['cG'].values[:10]], 
    'Predicted': [f"{val:.10f}" for val in y_pred[:10, 5]]
})
print(comparison)
