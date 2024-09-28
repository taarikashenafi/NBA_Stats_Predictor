from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib
from feature_engineering import preprocessed_data
import pandas as pd
import numpy as np

def train_xgboost_model(df, model_output_path="./models/xgboost_model.pkl", tune_hyperparameters=False):
    """
    Function to load data, train XGBoost model, and save the trained model.
    
    Parameters:
    - df: DataFrame, the preprocessed data.
    - model_output_path: str, path where the trained model will be saved.
    - tune_hyperparameters: bool, whether to perform hyperparameter tuning.

    Returns:
    - xgb_model: Trained XGBoost model.
    - X_test: Test features.
    - y_test: Test target values.
    """
    # Load preprocessed data and split into features and target
    X, y = preprocessed_data(df)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the base XGBoost model inside a MultiOutputRegressor for multi-target prediction
    base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    xgb_model = MultiOutputRegressor(base_model)

    # Hyperparameter tuning (if needed)
    if tune_hyperparameters:
        param_grid = {
            'estimator__max_depth': [3, 4, 5],
            'estimator__learning_rate': [0.01, 0.1, 0.2],
            'estimator__n_estimators': [100, 200, 300]
        }
        
        # Set up the GridSearchCV object
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        
        # Fit the model with hyperparameter tuning
        grid_search.fit(X_train, y_train)
        
        # Get the best model from the grid search
        xgb_model = grid_search.best_estimator_
        
        print(f"Best hyperparameters: {grid_search.best_params_}")
    else:
        # Train the model without hyperparameter tuning
        xgb_model.fit(X_train, y_train)
    
    # Evaluate on the test set
    y_pred = xgb_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred, multioutput='uniform_average')
    r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R2 Score: {r2:.4f}")
    
    # Save the trained model to disk
    joblib.dump(xgb_model, model_output_path)
    print(f"Model saved to {model_output_path}")
    
    # Return the trained model and the test set for further evaluation
    return xgb_model, X_test, y_test

# The following lines should be in a separate script, not in this function file
# if __name__ == "__main__":
#     nba_df = pd.read_csv("./data/NBA_Regular_Season_Stats_2021-2024.csv")
#     xgb_model, X_test, y_test = train_xgboost_model(nba_df, tune_hyperparameters=True)