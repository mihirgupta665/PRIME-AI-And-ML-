import streamlit as st
from youtube_analyzer import build_youtube_agent

import pandas as pd

st.set_page_config(page_title="Youtube Video Analyzer", layout="centered")

st.title("🎥 AI Youtube Video Analyzer")

def get_agent():
    return build_youtube_agent()

agent = get_agent
