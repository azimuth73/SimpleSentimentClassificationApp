from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode
import nltk

# Download NLTK resources if not already downloaded
nltk.download('stopwords')


def remove_punctuation(text):
    # TODO:
    #  Some punctuation like !?.,():;" should be replaced with space
    #  Others like ' should be replaced with nothing (so that certain english words don't become separated)
    #  Currently this function only removes these at the start or end and doesn't take into account which chars should
    #  be replaced and which should be removed
    # Replace characters in words that start or end with certain punctuation characters
    punctuation_chars = '''.,:;!?()'"'''

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
    # TODO:
    #  Need to filter some characters which appear 3 or more times because in english it's definitely a typo
    # Remove accents and convert to ASCII
    text = unidecode(text)
    if verbose:
        print(f'Remove accents and convert to ASCII\n\n\t\t{text}\n\n')

    # Convert to lowercase
    text = text.lower()
    if verbose:
        print(f'Convert to lowercase\n\n\t\t{text}\n\n')

    # TODO:
    #   Removing mentions, hashtags and links should come here, before removing punctuation

    # Replace characters in words that start or end with certain punctuation characters
    text = remove_punctuation(text)
    if verbose:
        print(f'Replace characters in words that start or end with certain punctuation characters\n\n\t\t{text}\n\n')

    # Remove any words which contain non-letter characters anywhere inside them
    text = remove_non_letter_words(text)
    if verbose:
        print(f'Remove any words which contain non-letter characters anywhere inside them\n\n\t\t{text}\n\n')

    # Remove words with one letter
    text = ' '.join([word for word in text.split() if len(word) >= 2])
    if verbose:
        print(f'Remove words with one letter\n\n\t\t{text}\n\n')

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
    text_to_preprocess = f'''
    Hello, Peter!I am writing to you for one reason,btw it's really important, anyhow you should totallyyy look at the
    data I sent you earlier.You're gonna be rly happy about it,I'm sure!!!!! :) sooo here is the link for you 
    https://peter-data.com
    '''

    preprocess_text(text_to_preprocess, verbose=True)


if __name__ == '__main__':
    main()
