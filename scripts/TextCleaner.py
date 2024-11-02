import re
import string
from typing import List, Set, Optional, Union

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize


class TextCleaner:
    """A class for cleaning and preprocessing text data.

    Features:
    - Lowercase conversion
    - Punctuation removal
    - Stopword removal
    - Number removal/normalization
    - Extra whitespace removal
    - Special character removal
    - Lemmatization/Stemming
    - Custom token filtering
    """

    def __init__(
        self,
        raw_text: str,
        language: str = "english",
        remove_numbers: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        min_token_length: int = 2,
        custom_stopwords: Optional[Set[str]] = None,
        use_stemming: bool = False,
    ):
        """Initialize the TextCleaner with configuration options.

        Args:
            raw_text: Input text to be cleaned
            language: Language for stopwords (default: "english")
            remove_numbers: Whether to remove numerical tokens (default: True)
            remove_urls: Whether to remove URLs (default: True)
            remove_emails: Whether to remove email addresses (default: True)
            min_token_length: Minimum length for a token to be kept (default: 2)
            custom_stopwords: Additional stopwords to include (default: None)
            use_stemming: Whether to use stemming instead of lemmatization (default: False)
        """
        # Download required NLTK data if not already present
        try:
            nltk.data.find("tokenizers/punkt")
            nltk.data.find("corpora/stopwords")
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("punkt")
            nltk.download("stopwords")
            nltk.download("wordnet")
            nltk.download("averaged_perceptron_tagger")

        self.raw_input_text = raw_text
        self.language = language
        self.remove_numbers = remove_numbers
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.min_token_length = min_token_length
        self.use_stemming = use_stemming

        # Initialize stopwords
        self.stopwords_set = set(stopwords.words(language) + list(string.punctuation))
        if custom_stopwords:
            self.stopwords_set.update(custom_stopwords)

        # Initialize lemmatizer/stemmer
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer() if use_stemming else None

        # Compile regex patterns
        self.url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        self.email_pattern = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
        self.number_pattern = re.compile(r"\d+")
        self.special_char_pattern = re.compile(r"[^a-zA-Z\s]")

    def _preprocess_text(self, text: str) -> str:
        """Perform initial text preprocessing."""
        # Convert to lowercase
        text = text.lower()

        # Remove URLs if specified
        if self.remove_urls:
            text = self.url_pattern.sub("", text)

        # Remove email addresses if specified
        if self.remove_emails:
            text = self.email_pattern.sub("", text)

        # Remove numbers if specified
        if self.remove_numbers:
            text = self.number_pattern.sub("", text)

        return text

    def _normalize_token(self, token: str) -> str:
        """Normalize a single token using either lemmatization or stemming."""
        if self.use_stemming:
            return self.stemmer.stem(token)
        return self.lemmatizer.lemmatize(token)

    def clean_text(self, return_tokens: bool = False) -> Union[str, List[str]]:
        """Clean the input text based on the configured options.

        Args:
            return_tokens: If True, returns list of tokens instead of joined string

        Returns:
            Either a cleaned string or list of cleaned tokens

        Raises:
            ValueError: If the input text is empty or not a string
        """
        if not isinstance(self.raw_input_text, str) or not self.raw_input_text.strip():
            raise ValueError("Input text must be a non-empty string")

        # Initial preprocessing
        text = self._preprocess_text(self.raw_input_text)

        # Tokenize
        tokens = word_tokenize(text)

        # Clean and normalize tokens
        cleaned_tokens = []
        for token in tokens:
            # Skip if token is a stopword or punctuation
            if token in self.stopwords_set:
                continue

            # Skip if token is too short
            if len(token) < self.min_token_length:
                continue

            # Remove special characters
            token = self.special_char_pattern.sub("", token)
            if not token:  # Skip if token is empty after cleaning
                continue

            # Normalize token (lemmatize/stem)
            token = self._normalize_token(token)

            cleaned_tokens.append(token)

        if return_tokens:
            return cleaned_tokens

        return " ".join(cleaned_tokens)

    def get_sentences(self) -> List[str]:
        """Split the cleaned text into sentences.

        Returns:
            List of cleaned sentences
        """
        cleaned_text = self.clean_text()
        return sent_tokenize(cleaned_text)

    def get_word_frequencies(self) -> dict:
        """Get word frequencies from the cleaned text.

        Returns:
            Dictionary of word frequencies
        """
        tokens = self.clean_text(return_tokens=True)
        return {word: tokens.count(word) for word in set(tokens)}
