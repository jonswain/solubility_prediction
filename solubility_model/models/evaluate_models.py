"""Functions to evaluate models."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
from pycaret.regression import predict_model


def regression_models(descriptors, y_true, models):
    """Makes predictions from the given descriptions and evaluates how well these
    fit the true values.

    Args:
        descriptors (pandas dataframe): Pandas dataframe containing the features
        y_true (pandas series): Pandas series containing the true values
        models (list): A list of PyCaret models
        log (bool, optional): If the data used has been log transformed
    """
    names = [str(model["actual_estimator"]).split("(")[0] for model in models]
    all_data = pd.concat([y_true, descriptors], axis=1)
    train_predictions4 = [predict_model(x, all_data) for x in models]

    _, axis = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    for index, predictions in enumerate(train_predictions4):
        i = index // 2
        j = index % 2
        ax = axis[i][j]
        y_pred = predictions.prediction_label

        model_metrics = f"""
        R2: {r2_score(y_true, y_pred):.2f},
        MSE: {mean_squared_error(y_true, y_pred):.2f}
        RMSE: {(mean_squared_error(y_true, y_pred, squared=False)):.2f}
        MAE: {mean_absolute_error(y_true, y_pred):.2f}"""

        ax.scatter(y_true, y_pred)
        ax.set_title(f"{index+1}: {names[index]}")
        ax_min = min(pd.concat([y_true, y_pred], axis=0))
        ax_max = max(pd.concat([y_true, y_pred], axis=0))
        ax_range = ax_max - ax_min
        x = np.linspace(ax_min - (ax_range / 10), ax_max + (ax_range / 10), 1000)
        ax.plot(x, x, "--k")
        ax.set_xlim(ax_min - (ax_range / 10), ax_max + (ax_range / 10))
        ax.set_ylim(ax_min - (ax_range / 10), ax_max + (ax_range / 10))
        ax.set_xlabel("Experimental values")
        ax.set_ylabel("Predicted values")
        plt.text(0.1, 0.75, f"{model_metrics}", fontsize=15, transform=ax.transAxes)

    plt.show()


def regression_model(y_true, y_pred):
    """Displays a plot showing the performance of a model

    Inputs:
        y_true: True values for target
        y_pred: Predicted target values from model
    Output: Displays plot
    """
    model_metrics = f"""
    R2: {r2_score(y_true, y_pred):.2f},
    MSE: {mean_squared_error(y_true, y_pred):.2f}
    RMSE: {(mean_squared_error(y_true, y_pred, squared=False)):.2f}
    MAE: {mean_absolute_error(y_true, y_pred):.2f}"""

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    plt.scatter(y_true, y_pred)
    plt.title("Predicted vs Experimental values")
    ax_min = min(pd.concat([y_true, y_pred], axis=0))
    ax_max = max(pd.concat([y_true, y_pred], axis=0))
    ax_range = ax_max - ax_min
    plt.xlim(ax_min - (ax_range / 10), ax_max + (ax_range / 10))
    plt.ylim(ax_min - (ax_range / 10), ax_max + (ax_range / 10))
    x = np.linspace(ax_min - (ax_range / 10), ax_max + (ax_range / 10), 1000)
    plt.plot(x, x, "--k")
    plt.xlabel("Experimental values")
    plt.ylabel("Predicted values")
    plt.text(0.1, 0.75, f"{model_metrics}", fontsize=15, transform=ax.transAxes)
    plt.show()
