import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title='Home \u00B7 Simple Sentiment Classification')

st.text_input('Input text:', key='input_text')  # Stored in st.session_state.input_text

model_option = ['Model 1', 'Model 2', 'Model 3', 'Model 4']
chosen_model = st.selectbox(
    'Select model:',
    model_option
)
