import streamlit as st

st.set_page_config(page_title='About \u00B7 Simple Sentiment Classification')

repository_link = "https://github.com/azimuth73/SimpleSentimentClassificationApp"

st.write(f'''
# Simple Sentiment Classification
This app implements machine learning based sentiment classification. Users can input text and select 
a model from the dropdown menu to evaluate the sentiment of the text. For more information about the specifics of
each model check out the [GitHub repository]({repository_link}) for this project.
''')
