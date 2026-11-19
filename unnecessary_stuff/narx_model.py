import sys
sys.path.append("src/models")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import make_windows, evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

# load data
d = np.load("data/aprbs_data.npz")
U = d["U"]
y = d["y"]

# Lag 0: only current state and input, no history
# is proven best by some experiments, in optimal_lag.py, methode taken from a paper

# Create features and targets
X, w = make_windows(y, U, L=0)

# Split into training and testing sets
Xtrain, Xtest, wtrain, wtest = train_test_split(X, w, test_size=0.2, random_state=42)

# Calculate Correllation Coefficient
corr_coef = np.corrcoef(np.concatenate([Xtest,wtest[:Xtest.shape[0]]], axis=1).T)

name_list = ['cA(t)','cB(t)','cC(t)', 'cP(t)','cE(t)','cG(t)','QB(t)','Tr(t)','cA(t+1)','cB(t+1)','cC(t+1)', 'cP(t+1)','cE(t+1)','cG(t+1)']
plt.figure(figsize=(10, 5))
sns.heatmap(corr_coef, xticklabels=name_list, yticklabels=name_list,cmap='coolwarm')
plt.show()


# Preprocess the data
scaler = StandardScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)
Xtest_scaled = scaler.transform(Xtest)

# Train Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(Xtrain_scaled, wtrain)

# Save model
joblib.dump({"model": lin_reg, "scaler": scaler}, "models/lin_reg_1step.joblib")

# Evaluate the model
evaluate_model(lin_reg, Xtrain_scaled, wtrain, Xtest_scaled, wtest)

# compare with trivial model
y_now = Xtest[:, :6]
nrmse_persist = np.sqrt(np.mean((wtest - y_now)**2, axis=0)) / wtest.std(axis=0)
print('trivial model (new state = old state):\nNRMSE:', nrmse_persist)
r2_persist_mean = r2_score(wtest, y_now)
print('R^2 Score:', r2_persist_mean,'\n')


# Train Random Forest model
forest_reg = RandomForestRegressor()
forest_reg.fit(Xtrain_scaled, wtrain)

# Save model
joblib.dump({"model": forest_reg, "scaler": scaler}, "models/forest_reg_1step.joblib")

# Evaluate the Random Forest model
evaluate_model(forest_reg, Xtrain_scaled, wtrain, Xtest_scaled, wtest)

print('Feature Importances:', forest_reg.feature_importances_)


# Train new multiple random forest model for each target variable
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

multi_reg = MultiOutputRegressor(RandomForestRegressor(random_state=0))
multi_reg.fit(Xtrain_scaled, wtrain)

# Save model
joblib.dump({"model": multi_reg, "scaler": scaler}, "models/multi_reg_1step.joblib")

evaluate_model(multi_reg, Xtrain_scaled, wtrain, Xtest_scaled, wtest)

names = ['cA','cB','cC','cP','cE','cG','QB','Tr']
targets = ['cA','cB','cC','cP','cE','cG']
for tname, est in zip(targets, multi_reg.estimators_):
    imp = est.feature_importances_.round(3).tolist()
    print(tname, dict(zip(names, imp)))


print('Conclusion: So far the model only learns next state=past state\n, no dynamic or imput dependece is learned.')

# let model learn the difference between states instead

