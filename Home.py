import streamlit as st
import numpy as np
import pandas as pd
import random

REPOSITORY_LINK = "https://github.com/azimuth73/SimpleSentimentClassificationApp"

MODEL_OPTION_NAMES = ['Model 1', 'Model 2', 'Model 3', 'Model 4']

NEGATIVE_EMOJI_SHORTCODE = ':slightly_frowning_face:'
NEUTRAL_EMOJI_SHORTCODE = ':neutral_face:'
POSITIVE_EMOJI_SHORTCODE = ':slightly_smiling_face:'

INPUT_TEXT_AREA_DESC = 'The text that will be used as an input to the selected model for sentiment classification.'
MODEL_SELECT_BOX_DESC = f'''
For more information about the specifics of
each model check out the [GitHub repository]({REPOSITORY_LINK}) for this project.
'''
EVAL_BUTTON_DESC = 'Evaluate the sentiment of the written text using the selected model.'

st.set_page_config(page_title='Home \u00B7 Simple Sentiment Classification')

if 'eval_button_clicked' not in st.session_state:  # Stateful Button
    st.session_state.eval_button_clicked = False

if 'eval_text' not in st.session_state:
    st.session_state.eval_text = ''

if 'eval_model_index' not in st.session_state:
    st.session_state.eval_model_index = 0

if 'current_input_text' not in st.session_state:
    st.session_state.current_input_text = ''


def eval_button_func(input_text: str, model_name: str) -> None:
    # TODO: Check if preprocessed text is not null or too simple in the future
    if not input_text:  # If string is empty ; later this should "do more" - check for whitespaces, special chars etc.
        st.warning('Input text field cannot be empty!')
        st.session_state.eval_button_clicked = False
        return

    if not st.session_state.eval_button_clicked:  # Stateful Button
        st.session_state.eval_button_clicked = True

    st.session_state.eval_text = input_text
    st.session_state.eval_model_name = model_name

    # TODO: actually implement the chosen model making a prediction
    st.session_state.eval_score = random.choice([-1, 0, 1])  # Dummy implementation

    if st.session_state.eval_score > 0:
        st.session_state.eval_emoji_shortcode = POSITIVE_EMOJI_SHORTCODE
    elif st.session_state.eval_score < 0:
        st.session_state.eval_emoji_shortcode = NEGATIVE_EMOJI_SHORTCODE
    else:
        st.session_state.eval_emoji_shortcode = NEUTRAL_EMOJI_SHORTCODE


st.text_area(  # Stored in st.session_state.input_text_area
    label='Input text:',
    value=st.session_state.current_input_text,
    key='input_text_area',
    help=INPUT_TEXT_AREA_DESC
)
st.session_state.current_input_text = st.session_state.input_text_area  # Temp to set the value while not initialised


selected_model_name = st.selectbox(
    label='Select model:',
    options=MODEL_OPTION_NAMES,
    index=st.session_state.eval_model_index,
    key='model_select_box',
    help=MODEL_SELECT_BOX_DESC
)
st.session_state.eval_model_index = MODEL_OPTION_NAMES.index(selected_model_name)

eval_func_args = (st.session_state.input_text_area, selected_model_name)  # Can use either temp or input_text_area
st.button(
    label='Evaluate', key='eval_button', help=EVAL_BUTTON_DESC,
    on_click=eval_button_func, args=eval_func_args, kwargs=None,
    type="secondary", use_container_width=True
)

# TODO: Make proper output based on the prediction of the chosen model
if st.session_state.eval_button_clicked:  # Dummy output display
    st.write(f'''
    {st.session_state.eval_text}
    {st.session_state.eval_model_name}
    {st.session_state.eval_score}
    {st.session_state.eval_emoji_shortcode}
    ''')
