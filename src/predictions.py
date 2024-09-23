import numpy as np
from model_training import train_xgboost_model

def predict_points(file_path, input_data):
    # Load the trained model
    xgb_model, _, _ = train_xgboost_model(file_path)
    
    # Make prediction based on user input
    input_data = np.array([input_data])
    prediction = xgb_model.predict(input_data)
    return prediction