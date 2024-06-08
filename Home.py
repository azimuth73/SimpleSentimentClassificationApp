import streamlit as st
import os
from utils import (
    predict_sentiment,
    download_and_load_model_folders_json,
    temporarily_download_and_load_model_files,
    load_image, load_metrics, load_model
)

# Constants
REPOSITORY_LINK = "https://github.com/azimuth73/SimpleSentimentClassificationApp"
NEGATIVE_EMOJI_SHORTCODE = ':slightly_frowning_face:'
POSITIVE_EMOJI_SHORTCODE = ':slightly_smiling_face:'
INPUT_TEXT_AREA_DESC = 'The text that will be used as an input to the selected model for sentiment classification.'
MODEL_SELECT_BOX_DESC = f'''
For more information about the specifics of
each model check out the [GitHub repository]({REPOSITORY_LINK}) for this project.
'''
EVAL_BUTTON_DESC = 'Evaluate the sentiment of the written text using the selected model.'

# Streamlit Config
st.set_page_config(page_title='Home \u00B7 Simple Sentiment Classification')

# Session State Initialization
if 'is_model_downloading' not in st.session_state:
    st.session_state.is_model_downloading = False
if 'eval_model_index' not in st.session_state:
    st.session_state.eval_model_index = 0
if 'model_folders_json' not in st.session_state:
    st.session_state.model_folders_json = download_and_load_model_folders_json()
if 'model_option_names' not in st.session_state:
    st.session_state.model_option_names = [model_folder['name'] for model_folder in st.session_state.model_folders_json]
if 'eval_button_clicked' not in st.session_state:
    st.session_state.eval_button_clicked = False
if 'eval_text' not in st.session_state:
    st.session_state.eval_text = ''
if 'current_input_text' not in st.session_state:
    st.session_state.current_input_text = ''
if 'model' not in st.session_state:
    st.session_state.model = None


# Download Model Files with Caching
@st.cache_resource
def download_model_files(eval_model_index):
    st.session_state.is_model_downloading = True
    files = temporarily_download_and_load_model_files(st.session_state.model_option_names[eval_model_index])
    st.session_state.is_model_downloading = False
    return files


def on_selectbox_change():
    st.session_state.eval_button_clicked = False


def eval_button_func(input_text: str, model_name: str) -> None:
    if not input_text:
        st.warning('Input text field cannot be empty!')
        st.session_state.eval_button_clicked = False
        return

    st.session_state.eval_button_clicked = True
    st.session_state.eval_text = input_text
    st.session_state.eval_model_name = model_name
    st.session_state.eval_score = predict_sentiment(st.session_state.model, st.session_state.eval_text)

    if st.session_state.eval_score == 1:
        st.session_state.eval_emoji_shortcode = POSITIVE_EMOJI_SHORTCODE
    else:
        st.session_state.eval_emoji_shortcode = NEGATIVE_EMOJI_SHORTCODE


# Main UI Elements
if st.session_state.is_model_downloading:
    st.write('Please wait while the model is downloading. This may take some time...')
else:
    st.text_area(
        label='Input text:',
        value=st.session_state.current_input_text,
        key='input_text_area',
        help=INPUT_TEXT_AREA_DESC
    )
    st.session_state.current_input_text = st.session_state.input_text_area

    selected_model_name = st.selectbox(
        label='Select model:',
        options=st.session_state.model_option_names,
        index=st.session_state.eval_model_index,
        key='model_select_box',
        help=MODEL_SELECT_BOX_DESC,
        on_change=on_selectbox_change
    )
    st.session_state.eval_model_index = st.session_state.model_option_names.index(selected_model_name)

    # Make sure to load the model only if it's not already loaded or if the model index has changed
    if st.session_state.model is None or st.session_state.eval_model_index != st.session_state.prev_eval_model_index:
        model_files = download_model_files(st.session_state.eval_model_index)
        st.session_state.model, st.session_state.metrics, st.session_state.roc_curve, st.session_state.confusion_matrix = model_files
        st.session_state.prev_eval_model_index = st.session_state.eval_model_index

    st.button(
        label='Evaluate Sentiment', key='eval_button', help=EVAL_BUTTON_DESC,
        on_click=eval_button_func, args=(st.session_state.input_text_area, selected_model_name),
        type='secondary', use_container_width=True
    )

    # Display results
    if st.session_state.eval_button_clicked and 'eval_score' in st.session_state:
        col1, col2, col3 = st.columns(3)
        col1_container = col1.container(border=True)
        col2_container = col2.container(border=True)
        col3_container = col3.container(border=True)
        col1_container.write(st.session_state.eval_score)
        col2_container.write('POSITIVE SENTIMENT' if st.session_state.eval_score == 1 else 'NEGATIVE SENTIMENT')
        col3_container.write(st.session_state.eval_emoji_shortcode)

    if 'metrics' in st.session_state:
        scores = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'Loss']
        cols = st.columns(6)
        for i, col in enumerate(cols):
            tile = col.container(border=True)
            tile.metric(label=scores[i], value=f'{st.session_state.metrics.get(scores[i], 0):.2f}')

    if 'confusion_matrix' in st.session_state and 'roc_curve' in st.session_state:
        cm_col, rc_col = st.columns(2)
        cm_container = cm_col.container(border=True)
        rc_container = rc_col.container(border=True)
        cm_container.image(st.session_state.confusion_matrix, caption='Confusion matrix of test data')
        rc_container.image(st.session_state.roc_curve, caption='ROC curve of test data')
