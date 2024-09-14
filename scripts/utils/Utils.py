import re
from uuid import uuid4

import spacy

# Load the English model
nlp = spacy.load("en_core_web_md")

REGEX_PATTERNS = {
    "email_pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "phone_pattern": r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
    "link_pattern": r"\b(?:https?://|www\.)\S+\b",
}


def generate_unique_id():
    """
    Generate a unique ID and return it as a string.

    Returns:
        str: A string with a unique UUID.
    """
    return str(uuid4())


class TextCleaner:
    """
    A class for cleaning text by removing specific patterns such as emails, links,
    punctuation, and stopwords, and for normalizing text.
    """

    def remove_emails_links(self, text: str) -> str:
        """
        Remove emails and links from the input text using regex patterns.

        Args:
            text (str): The input text to clean.

        Returns:
            str: The text with emails and links removed.
        """
        for pattern in REGEX_PATTERNS.values():
            text = re.sub(pattern, "", text)
        return text

    def remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation from the input text.

        Args:
            text (str): The input text to clean.

        Returns:
            str: The text with punctuation removed.
        """
        doc = nlp(text)
        tokens = [
            token.text for token in doc if not token.is_punct and not token.is_space
        ]
        return " ".join(tokens)

    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from the input text.

        Args:
            text (str): The input text to clean.

        Returns:
            str: The text with stopwords removed.
        """
        doc = nlp(text)
        tokens = [
            token.text
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space
        ]
        return " ".join(tokens)

    def normalize_text(self, text: str) -> str:
        """
        Normalize the text by converting to lowercase and removing extra whitespace.

        Args:
            text (str): The input text to normalize.

        Returns:
            str: The normalized text.
        """
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def clean_text(self, text: str) -> str:
        """
        Clean the input text by removing emails, links, punctuation, stopwords,
        and normalizing it.

        Args:
            text (str): The input text to clean.

        Returns:
            str: The fully cleaned text.
        """
        text = self.remove_emails_links(text)
        text = self.remove_punctuation(text)
        text = self.remove_stopwords(text)
        text = self.normalize_text(text)
        return text


class CountFrequency:
    """
    A class for counting the frequency of parts of speech or words in text.
    """

    def __init__(self, text: str):
        """
        Initialize the CountFrequency class.

        Args:
            text (str): The text to analyze.
        """
        self.text = text
        self.doc = nlp(text)

    def count_pos_frequency(self) -> dict:
        """
        Count the frequency of parts of speech in the input text.

        Returns:
            dict: A dictionary with POS tags as keys and their frequencies as values.
        """
        pos_freq = {}
        for token in self.doc:
            pos = token.pos_
            pos_freq[pos] = pos_freq.get(pos, 0) + 1
        return pos_freq

    def count_word_frequency(self) -> dict:
        """
        Count the frequency of words in the input text.

        Returns:
            dict: A dictionary with words as keys and their frequencies as values.
        """
        word_freq = {}
        for token in self.doc:
            if not token.is_punct and not token.is_space:
                word = token.text.lower()
                word_freq[word] = word_freq.get(word, 0) + 1
        return word_freq
