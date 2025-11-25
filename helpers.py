import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import  mean_squared_error, mean_absolute_error

from NeuralNet import NeuralNet

CSV_PATH = "model_results.csv"


def create_csv():
    if os.path.exists(CSV_PATH):
        results_df = pd.read_csv(CSV_PATH)
    else:
        results_df = pd.DataFrame(columns=['model', 'mse', 'mae', 'mape'])

    return results_df
    

def denormalize(x_norm,  y_train):
    y_min = y_train.min()
    y_max = y_train.max()
    return y_min + ((x_norm - 0.1) / 0.8) * (y_max - y_min)

def save_prediction_to_csv(path = CSV_PATH, metrics = {}):
    df = pd.DataFrame.from_dict(metrics, orient='index')
    df.index.name = 'model'
    df.reset_index(inplace=True)

    df.to_csv(path, index=False)

def get_prediction(name, model, X_test, y_test, y_train, metrics, show_prediction_vs_real = False):
    # Model prediction (still normalized)
    y_pred_norm = model.predict(X_test)
    
    # Denormalize predictions and real values
    y_pred = denormalize(y_pred_norm, y_train)
    y_true = denormalize(y_test, y_train)

    if show_prediction_vs_real:
        show_prediction_vs_real_values(y_pred, y_true, y_test)
    
    # Metrics
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Print rounded values
    print("MSE:", round(mse, 6))
    print("MAE:", round(mae, 6))
    print("MAPE:", round(mape, 6))

    # Dictionary used for saving
    metrics[name] = {
        'mse':  round(mse, 6),
        'mae':  round(mae, 6),
        'mape': round(mape, 6),
    }


def show_prediction_vs_real_values(y_pred, y_true, y_test):
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Real values')
    plt.ylabel('Predicted values')
    plt.title('Real vs Predicted values')
    plt.show()