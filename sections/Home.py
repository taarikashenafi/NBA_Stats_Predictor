#Home.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
def home():
    st.title("NBA Stat Predictor ")
    st.write("Welcome to the NBA Stat Predictor. To view predictions, please select a page from the sidebar.")

    @st.cache_data
    def load_data():
        nba_combined_df = pd.read_csv('./data/NBA_Regular_Season_Stats_2021-2024.csv')
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

            # Stats visualization
            stats = ['pts', 'ast', 'trb', 'fg%', '3p%']
            for stat in stats:
                st.markdown({stat})
                st.bar_chart(filtered_data.groupby('player')[stat].mean())
        else:
            st.write("Select at least two players to compare their stats.")

    def player_trend():
        st.markdown("## 📈 Player Performance Trends")
    
        # Load dataset
        data = load_data()

        season_order = ['2021-2022 Regular', '2022-2023 Regular', '2023-2024 Regular']

        data['season'] = pd.Categorical(data['season'], categories=season_order, ordered=True)
        
        # Select a player for trend analysis
        # Select a player for trend analysis
        player = st.selectbox('Select a player', ['Select a player...'] + list(data['player'].unique()))
        
        if player:
            # Filter the data for the selected player
            player_data = data[data['player'] == player]
            
            # Show performance trends for points
            st.markdown("### Points")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=player_data['season'], y=player_data['pts'], mode='markers+lines', marker=dict(size=10)))
            st.plotly_chart(fig)
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

    def current_leaders():
        st.markdown("## 🏀 NBA Stat Leaders")

        # Load data
        data = load_data()

        # Filter for the 2023-2024 season
        data = data[(data['season'] == '2023-2024 Regular') & (data['g'] >= 58)]



        # Create tabs for different stat categories
        tabs = st.tabs(["Points", "Assists", "Rebounds", "FG%", "3P%", "FT%", "Blocks", "Steals"])

        # Points tab
        with tabs[0]:
            top_scorers = data.sort_values(by="pts", ascending=False).head(10)
            top_scorers = top_scorers.reset_index(drop=True)
            top_scorers.index = top_scorers.index + 1
            st.markdown("### Top 10 Scorers")
            st.write(top_scorers[['player', 'pts', 'season']])
        
        # Assists tab
        with tabs[1]:
            top_assists = data.sort_values(by="ast", ascending=False).head(10)
            top_assists = top_assists.reset_index(drop=True)
            top_assists.index = top_assists.index + 1
            st.markdown("### Top 10 Assist Leaders")
            st.write(top_assists[['player', 'ast', 'season']])
        
        # Rebounds tab
        with tabs[2]:
            top_rebounders = data.sort_values(by="trb", ascending=False).head(10)
            top_rebounders = top_rebounders.reset_index(drop=True)
            top_rebounders.index = top_rebounders.index + 1
            st.markdown("### Top 10 Rebounders")
            st.write(top_rebounders[['player', 'trb', 'season']])
        
        # FG% tab
        with tabs[3]:
            top_fg = data.sort_values(by="fg%", ascending=False).head(10)
            top_fg = top_fg.reset_index(drop=True)
            top_fg.index = top_fg.index + 1
            st.markdown("### Top 10 FG%")
            top_fg['fg%'] = (top_fg['fg%'] * 100).round(2)
            st.write(top_fg[['player', 'fg%', 'season']])
        
        # 3P% tab
        with tabs[4]:
            top_3p = data.sort_values(by="3p%", ascending=False).head(10)
            top_3p = top_3p.reset_index(drop=True)
            top_3p.index = top_3p.index + 1
            st.markdown("### Top 10 3P%")
            top_3p['3p%'] = (top_3p['3p%'] * 100).round(2)
            st.write(top_3p[['player', '3p%', 'season']])
        
        # FT% tab
        with tabs[5]:
            top_ft = data.sort_values(by="ft%", ascending=False).head(10)
            top_ft = top_ft.reset_index(drop=True)
            top_ft.index = top_ft.index + 1
            st.markdown("### Top 10 FT%")
            top_ft['ft%'] = (top_ft['ft%'] * 100).round(2)
            st.write(top_ft[['player', 'ft%', 'season']])
        
        # Blocks tab
        with tabs[6]:
            top_blk = data.sort_values(by="blk", ascending=False).head(10)
            top_blk = top_blk.reset_index(drop=True)
            top_blk.index = top_blk.index + 1
            st.markdown("### Top 10 Block Leaders")
            st.write(top_blk[['player', 'blk', 'season']])
        
        # Steals tab
        with tabs[7]:
            top_stl = data.sort_values(by="stl", ascending=False).head(10)
            top_stl = top_stl.reset_index(drop=True)
            top_stl.index = top_stl.index + 1
            st.markdown("### Top 10 Steal Leaders")
            st.write(top_stl[['player', 'stl', 'season']])    
            
    player_comparison()
    current_leaders()
    player_trend()
