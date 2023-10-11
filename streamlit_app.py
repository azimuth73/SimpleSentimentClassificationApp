import streamlit as st
import numpy as np
import pandas as pd

repository_link = "https://github.com/azimuth73/SimpleSentimentClassificationApp"

st.write(f'''
# Simple Sentiment Classification
This app implements machine learning based sentiment classification. Users can input text and select 
a model from the dropdown menu to evaluate the sentiment of the text. For more information about the specifics of
each model check out the [GitHub repository]({repository_link}) for this project.
''')

st.text_input('Input text:', key='input_text')  # Stored in st.session_state.input_text

model_option = ['Model 1', 'Model 2', 'Model 3', 'Model 4']
chosen_model = st.selectbox(
    'Select model:',
    model_option
)
