import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def evaluate_model(model_path, X_test, y_test):
    """
    Function to evaluate the trained XGBoost model on test data.
    
    Parameters:
    - model_path: str, path to the saved model (e.g., 'xgboost_model.pkl').
    - X_test: Test features.
    - y_test: True test target values.

    Returns:
    - metrics: dict, containing the evaluation metrics.
    """

    # Load the trained model
    model = joblib.load(model_path)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print the results
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

    # Store the metrics in a dictionary
    metrics = {
        'MSE': mse,
        'MAE': mae,
        'R2': r2
    }

    # Plot residuals
    plot_residuals(y_test, y_pred)

    return metrics

def plot_residuals(y_true, y_pred):
    """
    Function to plot the residuals (differences between actual and predicted values).
    
    Parameters:
    - y_true: True target values.
    - y_pred: Predicted target values.
    """

    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    
    # Plot residuals
    plt.scatter(y_pred, residuals, color='blue', alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()

def plot_actual_vs_predicted(y_true, y_pred):
    """
    Function to plot actual vs predicted values.
    
    Parameters:
    - y_true: True target values.
    - y_pred: Predicted target values.
    """

    plt.figure(figsize=(10, 6))
    
    # Plot actual vs predicted
    plt.scatter(y_true, y_pred, color='green', alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='red', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Plot')
    plt.show()