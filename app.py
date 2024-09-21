import streamlit as st
from sections.Awards import awards
from sections.StatLeaders import statLeaders
from sections.Home import home

# Sidebar with title and navigation to different pages
st.sidebar.title("Predictions")

# Sidebar options for navigation
page = st.sidebar.selectbox(
    "Go to", 
    ["Home", "Awards", "Stat Leaders"]
)

# Display content based on selected page
if page == "Home":
    home()
elif page == "Awards":
    awards()
elif page == "Stat Leaders":
    statLeaders()