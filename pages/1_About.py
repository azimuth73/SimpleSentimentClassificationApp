import streamlit as st

st.set_page_config(page_title='About \u00B7 Simple Sentiment Classification')

repository_link = "https://github.com/azimuth73/SimpleSentimentClassificationApp"

st.write(f'''
# Project Overview
This application leverages various pre-trained models to perform sentiment analysis, providing insights into the 
emotional tone of the input text.. Users can input text and select 
a model from the dropdown menu to evaluate the sentiment of the text. For more information about the specifics of
each model check out the [GitHub repository]({repository_link}) for this project.
''')

st.markdown("""
## Dataset
The model is trained on SST2 dataset, a subset of the larger Stanford Sentiment Treebank dataset, a well-known benchmark 
in the field of sentiment 
analysis. The SST dataset provides a comprehensive set of movie reviews with binary sentiment labels. 
You can learn more about the SST2 dataset [here](https://huggingface.co/datasets/stanfordnlp/sst2).

## Model Training
The model training involves several key steps:
- **Data Loading**: The data is loaded from TSV files, with columns for labels and sentences. 
- **Data Processing**: The sentences are tokenized, truncated and padded to a fixed length, and then converted to 
tensors. This ensures that the input data is properly formatted for training.
- **Model Selection**: Multiple pre-trained models from Hugging Face's model hub are utilized, including:
    - **BERT** (bert-base-uncased, bert-large-uncased)
    - **RoBERTa** (roberta-base, roberta-large)
    - **AlBERT** (albert-base-v1, albert-large-v1, albert-xlarge-v1, albert-xxlarge-v1, albert-base-v2, albert-large-v2,
     albert-xlarge-v2, albert-xxlarge-v2)
    - **DistilRoBERTa** (distilroberta-base)
    - **CamemBERT** (camembert-base)
    - **XLM-RoBERTa** (xlm-roberta-base, xlm-roberta-large)
    - **XLNet** (xlnet-base-cased, xlnet-large-cased)
- **Fine-Tuning**: Each model is fine-tuned on the training data with specific parameters, including:
    - Maximum sequence length: 50
    - Number of epochs: 10
    - Batch size: 32
    - Learning rate: 2e-5
    - Early stopping with a patience of 1 epoch

## Pre-trained Models Architecture
The application employs a variety of pre-trained models to ensure robust performance:
- **bert-base-uncased**: 12-layer, 768-hidden, 12-heads, 110M parameters. Trained on lower-cased English text.
- **bert-large-uncased**: 24-layer, 1024-hidden, 16-heads, 340M parameters. Trained on lower-cased English text.
- **bert-base-cased**: 12-layer, 768-hidden, 12-heads, 110M parameters. Trained on cased English text.
- **bert-large-cased**: 24-layer, 1024-hidden, 16-heads, 340M parameters. Trained on cased English text.
- **albert-base-v1**: 12 repeating layers, 128 embedding, 768-hidden, 12-heads, 11M parameters. ALBERT base model.
- **albert-large-v1**: 24 repeating layers, 128 embedding, 1024-hidden, 16-heads, 17M parameters. ALBERT large model.
- **albert-xlarge-v1**: 24 repeating layers, 128 embedding, 2048-hidden, 16-heads, 58M parameters. ALBERT xlarge model.
- **albert-xxlarge-v1**: 12 repeating layer, 128 embedding, 4096-hidden, 64-heads, 223M parameters. 
ALBERT xxlarge model.
- **albert-base-v2**: 12 repeating layers, 128 embedding, 768-hidden, 12-heads, 11M parameters. ALBERT base model with 
no dropout, additional training data and longer training.
- **albert-large-v2**: 24 repeating layers, 128 embedding, 1024-hidden, 16-heads, 17M parameters. ALBERT large model 
with no dropout, additional training data and longer training.
- **albert-xlarge-v2**: 24 repeating layers, 128 embedding, 2048-hidden, 16-heads, 58M parameters. ALBERT xlarge model 
with no dropout, additional training data and longer training.
- **albert-xxlarge-v2**: 12 repeating layer, 128 embedding, 4096-hidden, 64-heads, 223M parameters. ALBERT xxlarge model
 with no dropout, additional training data and longer training.
- **roberta-base**: 12-layer, 768-hidden, 12-heads, 125M parameters. RoBERTa using the BERT-base architecture.
- **roberta-large**: 24-layer, 1024-hidden, 16-heads, 355M parameters. RoBERTa using the BERT-large architecture.
- **distilroberta-base**: 6-layer, 768-hidden, 12-heads, 82M parameters. The DistilRoBERTa model distilled from the 
RoBERTa model roberta-base checkpoint.
- **camembert-base**: 12-layer, 768-hidden, 12-heads, 110M parameters. CamemBERT using the BERT-base architecture.
- **xlm-roberta-base**: 12-layers, 768-hidden-state, 3072 feed-forward hidden-state, 8-heads, 125M parameters. Trained 
on on 2.5 TB of newly created clean CommonCrawl data in 100 languages.
- **xlm-roberta-large**: 24-layers, 1027-hidden-state, 4096 feed-forward hidden-state, 16-heads, 355M parameters. 
Trained on 2.5 TB of newly created clean CommonCrawl data in 100 languages.
- **xlnet-base-cased**: 12-layer, 768-hidden, 12-heads, 110M parameters. XLNet English model.
- **xlnet-large-cased**: 24-layer, 1024-hidden, 16-heads, 340M parameters. XLNet Large English model
  
  

""")


