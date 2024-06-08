import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv
import gdown
import json
import os
import numpy as np
from PIL import Image
import streamlit as st


class TransformerBinarySequenceClassificator(nn.Module):
    def __init__(self, model_name: str, requires_grad: bool = True) -> None:
        super(TransformerBinarySequenceClassificator, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        for parameter in self.model.parameters():
            parameter.requires_grad = requires_grad

    def forward(self, batch_sequences, batch_sequence_masks, batch_sequence_segments, labels=None):
        outputs = self.model(
            input_ids=batch_sequences,
            attention_mask=batch_sequence_masks,
            token_type_ids=batch_sequence_segments,
            labels=labels
        )
        loss, logits = outputs.loss, outputs.logits
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities


def predict_sentiment(model, sentence: str, max_sequence_length: int = 50):
    tokens_sequence = model.tokenizer.tokenize(sentence)
    tokens_sequence = ['[CLS]'] + tokens_sequence
    if len(tokens_sequence) > max_sequence_length:
        tokens_sequence = tokens_sequence[:max_sequence_length]
    padding = [0] * (max_sequence_length - len(tokens_sequence))
    input_ids = model.tokenizer.convert_tokens_to_ids(tokens_sequence) + padding
    attention_mask = [1] * len(tokens_sequence) + padding
    token_type_ids = [0] * max_sequence_length

    input_ids = torch.tensor([input_ids]).type(torch.long)
    attention_mask = torch.tensor([attention_mask]).type(torch.long)
    token_type_ids = torch.tensor([token_type_ids]).type(torch.long)

    with torch.no_grad():
        _, _, probabilities = model(input_ids, attention_mask, token_type_ids)
    prediction = torch.argmax(probabilities, dim=1).item()

    return prediction


@st.cache_data
def download_and_load_model_folders_json():
    if not os.path.exists('model_folders.json'):
        load_dotenv()
        model_folders_json_id = os.getenv('MODEL_FOLDERS_JSON_ID')
        gdown.download(id=model_folders_json_id, output='model_folders.json')

    with open('model_folders.json', 'r') as f:
        model_folders_json = json.load(f)

    return model_folders_json


def load_metrics(filepath):
    data = {}
    with open(filepath, 'r') as file:
        for line in file:
            metric_name, metric_value = line.strip().split('\t')
            data[metric_name] = float(metric_value)
    return data


def load_image(filepath):
    image = Image.open(filepath).convert('RGB')
    return np.array(image)


@st.cache_resource
def load_model(name: str):

    filepath = os.path.join('models', name, 'model.pt')
    classificator = TransformerBinarySequenceClassificator(model_name=name, requires_grad=False)
    classificator.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
    classificator.eval()
    return classificator


def temporarily_download_and_load_model_files(model_name: str):
    model_folders_json = download_and_load_model_folders_json()
    download_path = f'models/{model_name}'

    if not os.path.exists(download_path):
        model_folder_id = next((item['id'] for item in model_folders_json if item['name'] == model_name), None)
        if not model_folder_id:
            return None

        os.makedirs(download_path, exist_ok=True)
        gdown.download_folder(id=model_folder_id, output=download_path, quiet=False)

    model = load_model(model_name)
    metrics = load_metrics(os.path.join(download_path, 'metrics.tsv'))
    roc_curve_image = load_image(os.path.join(download_path, 'roc_curve.png'))
    confusion_matrix_image = load_image(os.path.join(download_path, 'confusion_matrix.png'))

    return model, metrics, roc_curve_image, confusion_matrix_image
