import streamlit as st
import random
from utils import predict_sentiment, download_and_load_model_folders_json, temporarily_download_and_load_model_files

REPOSITORY_LINK = "https://github.com/azimuth73/SimpleSentimentClassificationApp"

NEGATIVE_EMOJI_SHORTCODE = ':slightly_frowning_face:'
POSITIVE_EMOJI_SHORTCODE = ':slightly_smiling_face:'

INPUT_TEXT_AREA_DESC = 'The text that will be used as an input to the selected model for sentiment classification.'
MODEL_SELECT_BOX_DESC = f'''
For more information about the specifics of
each model check out the [GitHub repository]({REPOSITORY_LINK}) for this project.
'''
EVAL_BUTTON_DESC = 'Evaluate the sentiment of the written text using the selected model.'

st.set_page_config(page_title='Home \u00B7 Simple Sentiment Classification')

if 'is_model_downloading' not in st.session_state:
    st.session_state.is_model_downloading = False

if 'eval_model_index' not in st.session_state:
    st.session_state.eval_model_index = 0

if 'model_folders_json' not in st.session_state:
    st.session_state.model_folders_json = download_and_load_model_folders_json()

if 'model_option_names' not in st.session_state:
    st.session_state.model_option_names = [model_folder['name'] for model_folder in st.session_state.model_folders_json]


@st.cache_resource
def download_model_files(eval_model_index):
    if not st.session_state.is_model_downloading:

        st.session_state.is_model_downloading = True

        model_files = temporarily_download_and_load_model_files(
            st.session_state.model_option_names[eval_model_index]
        )
        model, metrics, roc_curve, confusion_matrix = model_files
        st.session_state.model = model
        st.session_state.metrics = metrics
        st.session_state.roc_curve = roc_curve
        st.session_state.confusion_matrix = confusion_matrix

        st.session_state.is_model_downloading = False


if 'eval_button_clicked' not in st.session_state:  # Stateful Button
    st.session_state.eval_button_clicked = False

if 'eval_text' not in st.session_state:
    st.session_state.eval_text = ''

if 'current_input_text' not in st.session_state:
    st.session_state.current_input_text = ''

if 'model' not in st.session_state:
    download_model_files(st.session_state.eval_model_index)


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

    st.session_state.eval_score = predict_sentiment(st.session_state.model, st.session_state.eval_text)

    if st.session_state.eval_score == 1:
        st.session_state.eval_emoji_shortcode = POSITIVE_EMOJI_SHORTCODE
    elif st.session_state.eval_score == 0:
        st.session_state.eval_emoji_shortcode = NEGATIVE_EMOJI_SHORTCODE


st.text_area(  # Stored in st.session_state.input_text_area
    label='Input text:',
    value=st.session_state.current_input_text,
    key='input_text_area',
    help=INPUT_TEXT_AREA_DESC
)
st.session_state.current_input_text = st.session_state.input_text_area  # Temp to set the value while not initialised


selected_model_name = st.selectbox(
    label='Select model:',
    options=st.session_state.model_option_names,
    index=st.session_state.eval_model_index,
    key='model_select_box',
    help=MODEL_SELECT_BOX_DESC,
)
st.session_state.eval_model_index = st.session_state.model_option_names.index(selected_model_name)
download_model_files(st.session_state.eval_model_index)

eval_func_args = (st.session_state.input_text_area, selected_model_name)  # Can use either temp or input_text_area
if 'model' in st.session_state and not st.session_state.is_model_downloading:
    st.button(
        label='Evaluate', key='eval_button', help=EVAL_BUTTON_DESC,
        on_click=eval_button_func, args=eval_func_args, kwargs=None,
        type='secondary', use_container_width=True
    )

# TODO: Make proper output based on the prediction of the chosen model
if st.session_state.eval_button_clicked:  # Dummy output display
    st.write(f'''
    {st.session_state.eval_text}
    {st.session_state.eval_model_name}
    {st.session_state.eval_score}
    {st.session_state.eval_emoji_shortcode}
    ''')

if 'metrics' in st.session_state:
    scores = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'Loss']
    row1 = st.columns(3)
    row2 = st.columns(3)

    for i, col in enumerate(row1 + row2):
        tile = col.container(height=120)
        tile.write(f'{scores[i]}')
        tile.write(st.session_state.metrics[scores[i]])
