import streamlit as st
from sections.AwardPredictions import awards
from sections.StatPredictions import stat_predictions
from sections.Home import home

# Sidebar with title and navigation to different pages
st.sidebar.title("Predictions")

# Sidebar options for navigation
page = st.sidebar.selectbox(
    "Go to", 
    ["Home", "Awards", "Stat Predictions"]
)

# Display content based on selected page
if page == "Home":
    home()
elif page == "Awards":
    awards()
elif page == "Stat Predictions":
    stat_predictions()