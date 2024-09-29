import joblib
import pandas as pd
from src.feature_engineering import preprocessed_data

def load_model(model_path):
    """
    Load the trained XGBoost model from a file.
    
    Parameters:
    - model_path: str, the path to the saved model file.

    Returns:
    - model: the loaded model.
    """
    model = joblib.load(model_path)
    return model

def preprocess_input(input_df):
    """
    Preprocess the input data to match the training data format.
    
    Parameters:
    - input_df: DataFrame, the raw input data to be processed.
    
    Returns:
    - input_df: DataFrame, the processed input data.
    """
    # Use the same preprocessing function as in training
    X, _ = preprocessed_data(input_df)
    return X

def make_prediction(model, input_data):
    """
    Make a prediction using the trained model.
    
    Parameters:
    - model: the trained model.
    - input_data: a dictionary containing the input data for prediction.

    Returns:
    - predictions: the predicted values (for points, assists, rebounds, steals, blocks).
    """
    # Convert input data (dictionary) into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess the input data
    input_df = preprocess_input(input_df)

    # Make the prediction (returns 5 outputs: pts, ast, trb, stl, blk)
    predictions = model.predict(input_df)
    
    return predictions[0]  # Return the first (and only) prediction

def predict_for_streamlit(model_path, input_data):
    """
    Make a prediction and format the result for Streamlit.
    
    Parameters:
    - model_path: str, path to the saved model.
    - input_data: dict, the user-provided input data for prediction.

    Returns:
    - result: str, formatted prediction result.
    """
    # Load the model
    model = load_model(model_path)
    
    # Make the prediction
    predictions = make_prediction(model, input_data)
    
    # Format the result for all 5 categories
    result = (f"Predicted Stats:\n"
              f"Points: {predictions[0]:.2f}\n"
              f"Assists: {predictions[1]:.2f}\n"
              f"Rebounds: {predictions[2]:.2f}\n"
              f"Steals: {predictions[3]:.2f}\n"
              f"Blocks: {predictions[4]:.2f}")
        
    return result

