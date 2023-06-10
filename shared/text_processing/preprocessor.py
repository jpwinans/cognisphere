import re
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class Preprocessor:
    def __init__(self):
        # Download necessary NLTK data packages
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        try:
            nltk.data.find("corpora/wordnet.zip")
        except LookupError:
            nltk.download("wordnet")

    def clean_text(self, text: str) -> str:
        """
        Cleans the text by removing special characters, numbers, and unnecessary white spaces.
        """
        # Remove special characters and numbers
        cleaned_text = re.sub(r"[^a-zA-Z\s]", "", text)

        cleaned_text = cleaned_text.replace("uh", "").replace("um", "")

        # Remove unnecessary white spaces
        cleaned_text = " ".join(cleaned_text.split())

        return cleaned_text

    def normalize_text(self, text: str) -> str:
        """
        Normalizes the text by performing tasks like lowercasing and lemmatization.
        """
        # Lowercase the text
        text = text.lower()

        # Tokenize the text
        words = word_tokenize(text)

        # Initialize the lemmatizer
        lemmatizer = WordNetLemmatizer()

        # Lemmatize the words
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

        # Join the lemmatized words back into a string
        normalized_text = " ".join(lemmatized_words)

        return normalized_text

    def text_to_array(self, text: str) -> str:
        """
        Converts the text into an array from line breaks.
        """
        lines = text.split("\n")
        return [self.clean_text(line) for line in lines if line != ""]
    
    def remove_special_tokens(self, text: str) -> str:
        """
        Removes special tokens from the text.
        """
        tokens = ['<|endoftext|>']
        for token in tokens:
            text = text.replace(token, '')
        return text
