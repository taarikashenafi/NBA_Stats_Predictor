# sections/StatPredictions.py

import streamlit as st
import pandas as pd
from src.predictions import predict_for_streamlit


def stat_predictions():
    st.markdown("## Predict Season Averages")


    # Load dataset
    data = pd.read_csv('./data/preprocessed_data.csv')  # Load your dataset from a CSV file

    # Get unique players from the dataframe
    players = data['player'].unique()


    # Allow user to select a player
    selected_player = st.selectbox('Select a player', players)


    if selected_player:
        # Filter data for the selected player
        player_data = data[data['player'] == selected_player]

        # Prepare input data for prediction
        input_data = {
            "player": selected_player,
            "pos": player_data['pos'].iloc[0],
            "age": player_data['age'].iloc[0],
            "tm": player_data['tm'].iloc[0],
            "g": player_data['g'].iloc[0],
            "gs": player_data['gs'].iloc[0],
            "mp": player_data['mp'].iloc[0],
            "fg": player_data['fg'].iloc[0],
            "fga": player_data['fga'].iloc[0],
            "fg%": player_data['fg%'].iloc[0],
            "3p": player_data['3p'].iloc[0],
            "3pa": player_data['3pa'].iloc[0],
            "3p%": player_data['3p%'].iloc[0],
            "2p": player_data['2p'].iloc[0],
            "2pa": player_data['2pa'].iloc[0],
            "2p%": player_data['2p%'].iloc[0],
            "efg%": player_data['efg%'].iloc[0],
            "ft": player_data['ft'].iloc[0],
            "fta": player_data['fta'].iloc[0],
            "ft%": player_data['ft%'].iloc[0],
            "orb": player_data['orb'].iloc[0],
            "drb": player_data['drb'].iloc[0],
            "trb": player_data['trb'].iloc[0],
            "ast": player_data['ast'].iloc[0],
            "stl": player_data['stl'].iloc[0],
            "blk": player_data['blk'].iloc[0],
            "tov": player_data['tov'].iloc[0],
            "pf": player_data['pf'].iloc[0],
            "pts": player_data['pts'].iloc[0],
            "season": player_data['season'].iloc[0],
            "player_id": player_data['player_id'].iloc[0],
            "pts_improvement": player_data['pts_improvement'].iloc[0],
            "ast_improvement": player_data['ast_improvement'].iloc[0],
            "trb_improvement": player_data['trb_improvement'].iloc[0],
            "stl_improvement": player_data['stl_improvement'].iloc[0],
            "blk_improvement": player_data['blk_improvement'].iloc[0],
            "fg%_improvement": player_data['fg%_improvement'].iloc[0],
            "3p%_improvement": player_data['3p%_improvement'].iloc[0],
            "ft%_improvement": player_data['ft%_improvement'].iloc[0],
            "weighted_pts": player_data['weighted_pts'].iloc[0],
            "weighted_ast": player_data['weighted_ast'].iloc[0],
            "weighted_trb": player_data['weighted_trb'].iloc[0],
            "pts_positional_rank": player_data['pts_positional_rank'].iloc[0],
            "ast_positional_rank": player_data['ast_positional_rank'].iloc[0],
            "trb_positional_rank": player_data['trb_positional_rank'].iloc[0],
            "stl_positional_rank": player_data['stl_positional_rank'].iloc[0],
            "blk_positional_rank": player_data['blk_positional_rank'].iloc[0],
            "fg%_positional_rank": player_data['fg%_positional_rank'].iloc[0],
            "3p%_positional_rank": player_data['3p%_positional_rank'].iloc[0],
            "ft%_positional_rank": player_data['ft%_positional_rank'].iloc[0],
            "efficiency": player_data['efficiency'].iloc[0],
            "efficiency_rank": player_data['efficiency_rank'].iloc[0],
            "consistency_score": player_data['consistency_score'].iloc[0],
            "consistency_rank": player_data['consistency_rank'].iloc[0]
        }

        # Make prediction using the model
        prediction = predict_for_streamlit(model_path='./models/xgboost_model.pkl', input_data=input_data)


        # Display prediction results
        st.markdown("### Predicted Season Averages")
        st.write(prediction)


        # Visualize prediction results
        st.bar_chart(player_data[['pts', 'ast', 'trb', 'fg%', '3p%']].mean())


