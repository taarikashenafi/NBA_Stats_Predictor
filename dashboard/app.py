# File: dashboard/app.py

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import load_and_preprocess_data
from src.predictions import predict_stat_leaders, predict_awards
from src.analysis import get_player_trends, compare_players

app = Flask(__name__)

# Load and preprocess data
data = load_and_preprocess_data()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/stat_leaders')
def stat_leaders():
    predictions = predict_stat_leaders(data)
    return render_template('stat_leaders.html', predictions=predictions)

@app.route('/awards')
def awards():
    predictions = predict_awards(data)
    return render_template('awards.html', predictions=predictions)

@app.route('/player_trends/<player_name>')
def player_trends(player_name):
    trends = get_player_trends(data, player_name)
    return render_template('player_trends.html', player=player_name, trends=trends)

@app.route('/compare_players', methods=['GET', 'POST'])
def compare_players_route():
    if request.method == 'POST':
        player1 = request.form['player1']
        player2 = request.form['player2']
        comparison = compare_players(data, player1, player2)
        return render_template('player_comparison.html', comparison=comparison)
    return render_template('compare_players_form.html')

@app.route('/api/players')
def get_players():
    players = sorted(data['Player'].unique())
    return jsonify(players)

@app.route('/api/player_stats/<player_name>')
def get_player_stats(player_name):
    player_data = data[data['Player'] == player_name].to_dict(orient='records')
    return jsonify(player_data)

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)