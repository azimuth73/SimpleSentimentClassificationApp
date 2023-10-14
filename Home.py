import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title='Home \u00B7 Simple Sentiment Classification')

input_text_area_desc = 'The text that will be used as an input to the selected model for sentiment classification.'
st.text_area(
    label='Input text:',
    key='input_text_area',
    help=input_text_area_desc
)  # Stored in st.session_state.input_text_area


model_option = ['Model 1', 'Model 2', 'Model 3', 'Model 4']
repository_link = "https://github.com/azimuth73/SimpleSentimentClassificationApp"
model_select_box_desc = f'''
For more information about the specifics of
each model check out the [GitHub repository]({repository_link}) for this project.
'''
chosen_model = st.selectbox(
    label='Select model:',
    key='model_select_box',
    options=model_option,
    help=model_select_box_desc
)

eval_button_desc = 'Evaluate the sentiment of the written text using the selected model.'
st.button(
    label='Evaluate', key='evaluate_text_button', help=eval_button_desc,
    on_click=None, args=None, kwargs=None,
    type="secondary", use_container_width=True
)
