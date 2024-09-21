import streamlit as st
import pandas as pd
def home():
    st.title("NBA Stat Predictor Home")
    st.write("Welcome to the NBA Stat Predictor. To view predictions, please select a page from the sidebar.")

    @st.cache_data
    def load_data():
        nba_combined_df = pd.read_csv('./data/NBA_Combined_Stats__2021-2024_.csv')
        return nba_combined_df  # Replace with your actual data loading logic

    # Add a new page for player comparison
    def player_comparison():
        st.markdown("## 🔄 Compare NBA Players")
    
        # Load dataset
        data = load_data()
    
        # Allow users to select multiple players to compare
        players = st.multiselect('Select players to compare', data['player'].unique())
    
        if players:
            # Filter dataset for the selected players
            filtered_data = data[data['player'].isin(players)]
        
            # Display the comparison table
            st.write(filtered_data[['player', 'season', 'pts', 'ast', 'trb', 'fg%', '3p%']])
        
            # Optionally, add some stats visualization (bar chart example)
            st.bar_chart(filtered_data.groupby('player')['pts'].mean())  # Average points per player
        else:
            st.write("Select at least two players to compare their stats.")

    def player_trend():
        st.markdown("## 📈 Player Performance Trends")
    
        # Load dataset
        data = load_data()

        season_order = [
            '2021-2022 Regular', '2021-2022 Playoffs',
            '2022-2023 Regular', '2022-2023 Playoffs',
            '2023-2024 Regular', '2023-2024 Playoffs'
        ]

        data['season'] = pd.Categorical(data['season'], categories=season_order, ordered=True)
        
        # Select a player for trend analysis
        # Select a player for trend analysis
        player = st.selectbox('Select a player', ['Select a player...'] + list(data['player'].unique()))
        
        if player:
            # Filter the data for the selected player
            player_data = data[data['player'] == player]
            
            # Show performance trends for points
            st.markdown("### Points")
            st.line_chart(player_data[['season', 'pts']].set_index('season'))
            st.write(f"Average points per game for {player}: {player_data['pts'].mean():.2f}")

            # Show performance trends for assists
            st.markdown("### Assists")
            st.line_chart(player_data[['season', 'ast']].set_index('season'))
            st.write(f"Average assists per game for {player}: {player_data['ast'].mean():.2f}")

            # Show performance trends for rebounds
            st.markdown("### Rebounds")
            st.line_chart(player_data[['season', 'trb']].set_index('season'))
            st.write(f"Average rebounds per game for {player}: {player_data['trb'].mean():.2f}")
            

            # Show performance trends for field goal percentage
            st.markdown("### Field Goal Percentage")
            st.line_chart(player_data[['season', 'fg%']].apply(lambda x: x*100 if x.name == 'fg%' else x).set_index('season'))
            st.write(f"Average field goal percentage for {player}: {player_data['fg%'].mean():.2f}")

            # Show performance trends for 3-point percentage
            st.markdown("### 3-Point Percentage")
            st.line_chart(player_data[['season', '3p%']].apply(lambda x: x*100 if x.name == '3p%' else x).set_index('season'))
            st.write(f"Average 3-point percentage for {player}: {player_data['3p%'].mean():.2f}")
            
        else:
            st.write("Select a player to view their performance trends.")

    player_comparison()
    player_trend()
