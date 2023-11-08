import numpy as np
import pandas as pd
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Load your dataset into a DataFrame
data = pd.read_csv('your_dataset.csv')

# Preprocessing
data['text'] = data['text'].str.lower()  # Convert text to lowercase
data['text'] = data['text'].str.replace(r'[^a-zA-Z\s]', '', regex=True)  # Remove punctuation
data['text'] = data['text'].str.replace(r'\s+', ' ', regex=True)  # Remove extra white spaces

# Define the labels for sentiment (assuming you have labels 0, 1, 2)
label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
data['sentiment'] = data['sentiment'].map(label_mapping)

# Tokenization and padding
max_words = 10000  # Adjust as needed
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(data['text'])
X = tokenizer.texts_to_sequences(data['text'])
X = pad_sequences(X, padding='post', maxlen=100)  # Adjust maxlen as needed

# Encode sentiment labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['sentiment'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a simple LSTM model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=32, input_length=X.shape[1]))
model.add(LSTM(64))
model.add(Dense(3, activation='softmax'))  # 3 classes: negative, neutral, positive

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Save the model to a file
model.save('sentiment_model.h5')
