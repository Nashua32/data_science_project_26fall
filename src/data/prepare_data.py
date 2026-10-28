"""
Comments:
    - the resulting df/csv is pretty huge, is this the best way?
    - so far, lag is fixed 5, is this good?



    - are they called features? 
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "data_generic.csv"))
#print(file_path)
df = pd.read_csv(file_path)

LAG_STEPS = 5


#creates data that could be used for training by shifting the features
def create_mimo_lagged_data(df, features_cols, input_cols, lag_steps=5):
    """
    Creates lagged features for a Multi-Output NARX model,
    incorporating past lags of all species and all inputs.
    df: the dataframe we are transforming
    features_cols:
    input_cols:
    lag_steps: the number of past states we want to use to predict the next one
    """
    data = df.copy()
    feature_columns_past = []
    
    #generate lags for all features
    for feature in features_cols:
        for i in range(1, lag_steps + 1):
            col_name = f"{feature}_lag{i}"
            data[col_name] = data[feature].shift(i)
            feature_columns_past.append(col_name)
            
    #generate lags for exogenous inputs
    for in_col in input_cols:
        for i in range(1, lag_steps + 1):
            col_name = f"{in_col}_lag{i}"
            data[col_name] = data[in_col].shift(i)
            feature_columns_past.append(col_name)
            
    #drop NaN rows due to shifting
    data = data.dropna().reset_index(drop=True)
    
    #contains past history of everything
    X = data[feature_columns_past]
    
    #contains current concentrations of all features at time t
    y = data[features_cols]

    #TODO: maybe change this so that it can actually be used for a model to train
    return X, y, feature_columns_past
    #return data

# Example usage for all 6 species:
features = ["cA", "cB", "cC", "cP", "cE", "cG"]
features_cols = [f"{f}" for f in features]
input_features = ["QB", "Tr"]

"""
lagged_df = create_mimo_lagged_data(df, features_cols, input_features, LAG_STEPS)

#save the new dataframe as a csv file

file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", f"prepared_data_lag{LAG_STEPS}.csv"))

lagged_df.to_csv(file_path, index=False)
#print(lagged_df.head())
print(f"Written prepared data into the following file: {file_path}")
"""