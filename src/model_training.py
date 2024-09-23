from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib
from feature_engineering import load_and_preprocess_data

def train_xgboost_model(file_path, model_output_path="./models/xgboost_model.pkl", tune_hyperparameters=False):
    """
    Function to load data, train XGBoost model, and save the trained model.
    
    Parameters:
    - file_path: str, path to the preprocessed data.
    - model_output_path: str, path where the trained model will be saved.
    - tune_hyperparameters: bool, whether to perform hyperparameter tuning.

    Returns:
    - xgb_model: Trained XGBoost model.
    - X_test: Test features.
    - y_test: Test target values.
    """
    # Load preprocessed data
    X, y = load_and_preprocess_data(file_path)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the base XGBoost model
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    # Hyperparameter tuning
    if tune_hyperparameters:
        param_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200, 300]
        }
        
        # Set up the GridSearchCV object
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
        
        # Fit the model with hyperparameter tuning
        grid_search.fit(X_train, y_train)
        
        # Get the best model from the grid search
        xgb_model = grid_search.best_estimator_
        
        print(f"Best hyperparameters: {grid_search.best_params_}")
    
    # Train the model
    xgb_model.fit(X_train, y_train)
    
    # Evaluate on the test set (optional evaluation step)
    y_pred = xgb_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE: {mse:.4f}")
    
    # Save the trained model to disk
    joblib.dump(xgb_model, model_output_path)
    print(f"Model saved to {model_output_path}")
    
    # Return the trained model and the test set for further evaluation
    return xgb_model, X_test, y_test