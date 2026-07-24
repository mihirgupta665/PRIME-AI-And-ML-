import streamlit as st
from youtube_analyzer import build_youtube_agent

import pandas as pd

st.set_page_config(page_title="Youtube Video Analyzer", layout="centered")

st.title("🎥 AI Youtube Video Analyzer")

@st.cache_resource
def get_agent():
    return build_youtube_agent()

agent = get_agent()

# input box
video_url = st.text_input("Enter Youtube Video URL")
button = st.button("Analyse Video")
# print(video_url)
# print(button)

if video_url and button:
    with st.spinner("Analyzing Video...."):
        response = agent.run(
            f"Analyse this video: {video_url}"
        )

    # print(response)
    st.markdown("Analysis Report of Video:")
    st.markdown(response.content)   

