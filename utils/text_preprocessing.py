import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode
import nltk

# Download NLTK resources if not already downloaded
nltk.download('stopwords')


def preprocess_text(text):
    # Remove accents and convert to ASCII
    text = unidecode(text)
    # Convert to lowercase
    text = text.lower()
    # Remove words with non-letter characters
    text = re.sub(r'\b[^a-z]+\b', ' ', text)  # This regular expression doesnt quite work, need to find a replacement
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    return text


def main():
    # Example usage
    text_to_preprocess = "This is an example text with stopwords and some special characters! Visit https://example.com for more info."
    preprocessed_text = preprocess_text(text_to_preprocess)
    print(preprocessed_text)


if __name__ == '__main__':
    main()
