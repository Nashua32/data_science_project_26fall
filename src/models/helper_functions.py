import numpy as np


def make_windows(y, U, L):
    t = np.arange(L, len(U))
    y_hist = [y[t - j] for j in range(L + 1)]
    u_hist = [U[t - j] for j in range(L + 1)]
    X = np.hstack(y_hist + u_hist)
    W = y[t + 1]
    return X, W

# Normed (on std) Root Mean Square Error
def nrmse(model, X, y):
    y_pred = model.predict(X)
    rmse_each = np.sqrt(np.mean((y - y_pred)**2, axis=0))
    return rmse_each / y.std(axis=0)

def evaluate_model(model, Xtrain, wtrain, Xtest, wtest):
    nrmse_train = nrmse(model, Xtrain, wtrain)
    nrmse_test = nrmse(model, Xtest, wtest)
    score = model.score(Xtest, wtest)

    print(model.__class__.__name__)
    print('Train NRMSE:', nrmse_train)
    print('Test NRMSE:', nrmse_test)
    print('Test R^2 Score:', score)
    print('')
    return 

def make_windows_multistep(y, U, L, H):
    N = len(U); 
    t = np.arange(L, N - H + 1)
    y_hist = [y[t - j] for j in range(L + 1)]
    u_past = [U[t - j] for j in range(1, L + 1)]
    u_fut  = [U[t + k] for k in range(H)]
    X = np.hstack(y_hist + u_past + u_fut)
    W = np.hstack([y[t + k] for k in range(1, H + 1)])
    return X, W