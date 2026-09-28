# Description of folders

## inside *src*

### src/simulation

Here we define the mathematical models that we can use for generating the data

### src/data

Here we use the files defined in src/simulation to actually generate data for our models that we then store in the data folder. We also transform data here in a way so that it can be used as input for the machine learning models

### src/models

Here we train the ML models and store them in the models folder

### src/evaluation 

Here we evaluate our results (probs details needed)

## inside *data*

Here we store the data that we have generated so that we can use it for training the ML models

## inside *models*
 
Here we store the trained ML models