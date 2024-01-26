import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode
import nltk

# Download NLTK resources if not already downloaded
nltk.download('stopwords')


def remove_punctuation(text):

    # Replace characters in words that start or end with certain punctuation characters
    punctuation_chars = '''.,:;!?()'"/#'''

    # strip method removes heading and trailing characters
    for char in punctuation_chars:
        text = ' '.join([word.strip(char) for word in text.split()])
    return text


def remove_non_letter_words(text):
    non_letter_chars = '''!"#$%&'()*+,-./0123456789:;<=>?@[\\]^_`{|}~'''

    words = text.split()

    # Create a new list with filtered words
    filtered_words = [word for word in words if not any(char in non_letter_chars for char in word)]

    # Join the filtered words back into a text string
    result_text = ' '.join(filtered_words)

    return result_text


def preprocess_text(text, verbose=False):
    # Remove accents and convert to ASCII
    text = unidecode(text)
    if verbose:
        print(f'Remove accents and convert to ASCII\n\n\t\t{text}\n\n')

    # Convert to lowercase
    text = text.lower()
    if verbose:
        print(f'Convert to lowercase\n\n\t\t{text}\n\n')

    # Replace characters in words that start or end with certain punctuation characters
    text = remove_punctuation(text)
    if verbose:
        print(f'Replace characters in words that start or end with certain punctuation characters\n\n\t\t{text}\n\n')

    # Remove any words which contain non-letter characters anywhere inside them
    text = remove_non_letter_words(text)
    if verbose:
        print(f'Remove any words which contain non-letter characters anywhere inside them\n\n\t\t{text}\n\n')

    # Remove words with two letters or fewer
    text = ' '.join([word for word in text.split() if len(word) >= 3])
    if verbose:
        print(f'Remove words with two letters or fewer\n\n\t\t{text}\n\n')

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    if verbose:
        print(f'Remove stopwords\n\n\t\t{text}\n\n')

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    if verbose:
        print(f'Lemmatization\n\n\t\t{text}\n\n')

    return text


def main():
    # Example usage:
    text_to_preprocess = "Aww Aw this is an example text with stopwords and some special characters! " \
                         "Visit https://example.com for more        info. #blessed #JustForLaughs " \
                         "I am so happy today man. :) This stuff is pretty crazy!!!!"
    preprocessed_text = preprocess_text(text_to_preprocess, verbose=True)


if __name__ == '__main__':
    main()
