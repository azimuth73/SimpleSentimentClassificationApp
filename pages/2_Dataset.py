import streamlit as st

st.set_page_config(page_title='Dataset \u00B7 Simple Sentiment Classification')

SOURCE_LINK_1 = 'https://www.kaggle.com/datasets/tariqsays/sentiment-dataset-with-1-million-tweets?resource=download'
SOURCE_LINK_2 = 'https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset'

st.write(f'''
## Source
The train and test datasets were filtered from these existing datasets found on Kaggle:
 - [Twitter dataset #1]({SOURCE_LINK_1})
 - [Twitter dataset #2]({SOURCE_LINK_2})
''')

st.write(f'''
## Preprocessing
The order of preprocessing methods for each entry in the dataset, as well as each input to the models made on this app 
is as follows:
 - Convert the text to ASCII
 - Convert the text to lowercase
 - Remove heading and trailing punctuation
 - Remove words which contain special characters
 - Remove words containing only one letter
 - Remove stop words found in the English language
 - Shorten words to their base form (Lemmatization)
''')
