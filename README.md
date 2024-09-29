# NBA Stats Predictor

This project utilizes machine learning to predict NBA player statistics for upcoming seasons. Built with a focus on real-world data processing and prediction accuracy, the app allows users to explore and visualize the performance predictions of NBA players. This project aims to demonstrate proficiency in data science, machine learning model development, and full-stack application deployment using modern technologies.

## Features

- **Player Stats Prediction:** Predicts key NBA player statistics such as points, assists, rebounds, and more for the upcoming season.
- **Data Processing & Feature Engineering:** Implements advanced feature engineering techniques to clean, preprocess, and enhance the dataset for improved model accuracy.
- **Model Training & Evaluation:** Trains machine learning models with multiple algorithms (XGBoost, Random Forest, etc.) and evaluates performance using metrics such as MSE.
- **Real-Time Data Visualization:** Utilizes Streamlit for real-time, interactive data visualization.
- **Historical Data Analysis:** Leverages historical data to make informed predictions and visualize trends.

## Technologies Used

- **Frontend:**
  - **Streamlit:** For creating an interactive dashboard to display predictions and visualize data.
  
- **Backend & Data Processing:**
  - **Python:** Core programming language for data processing and machine learning.
  - **Pandas & NumPy:** For data manipulation and mathematical operations.
  - **Scikit-learn:** For model training and evaluation.
  - **XGBoost:** Gradient boosting framework used for training highly accurate models.
  
- **Data Storage:**
  - **CSV Files:** Data loaded from cleaned and preprocessed CSV files for ease of access.

- **Other Tools:**
  - **Jupyter Notebooks:** For developing and testing various machine learning models.
  - **Git & GitHub:** Version control for tracking project changes and collaborations.
  - **Virtual Environment (.venv):** Isolated environment to manage dependencies.

## Machine Learning Workflow

1. **Data Collection:** Historical NBA data (points, assists, rebounds, etc.) is collected and cleaned to remove irrelevant or noisy data.
2. **Feature Engineering:** Important features such as player position, minutes played, team performance, and advanced statistics are extracted.
3. **Model Selection:** Various machine learning algorithms (e.g., XGBoost) are evaluated to select the best-performing model for predicting stats.
4. **Model Training:** The model is trained on past seasons' data to predict the statistics for the next season.
5. **Evaluation:** Model performance is evaluated using Mean Squared Error (MSE) and other relevant metrics.
6. **Deployment:** Predictions are visualized in real-time using Streamlit, allowing users to select a player and view their predicted performance.

## Installation

Clone the repository:

```bash
git clone https://github.com/taarikashenafi/NBA_Stats_Predictor.git
```

Navigate to the project directory:

```bash
cd NBA_Stats_Predictor
```
Create and activate the virtual environment (optional but recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

To run the app and view predictions:

```bash
streamlit run app.py
```
Navigate to the provided localhost link in your browser to interact with the dashboard.

## Data Sources

-	[2023-2024 NBA Player Stats](https://www.kaggle.com/datasets/vivovinco/2023-2024-nba-player-stats)
-	[2022-2021 NBA Player Stats](https://www.kaggle.com/datasets/vivovinco/20222023-nba-player-stats-regular)
-	[2021-2022 NBA Player Stats](https://www.kaggle.com/datasets/vivovinco/nba-player-stats)

## Upcoming Features

-	NBA Awards Predictions: Predictions for NBA awards like MVP, Defensive Player of the Year, and Rookie of the Year (under development).
- NBA Team Predictions: Predictions for team regualar season records and playoff brackets

## Contributions

Feel free to contribute by forking the repository and submitting pull requests. All feedback and contributions are welcome!



