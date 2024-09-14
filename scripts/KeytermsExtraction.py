import spacy
from textacy import extract


class KeytermExtractor:
    """
    A class for extracting key terms from a given text using various algorithms.
    """

    def __init__(self, raw_text: str, top_n_values: int = 20, language_model=None):
        """
        Initialize the KeytermExtractor object.

        Args:
            raw_text (str): The raw input text.
            top_n_values (int): The number of top key terms to extract.
            language_model (spacy.language.Language, optional): A preloaded spaCy language model.
                If None, 'en_core_web_sm' will be loaded by default.
        """
        self.raw_text = raw_text
        self.top_n_values = top_n_values
        if language_model is None:
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = language_model
        self.text_doc = self.nlp(self.raw_text)

    def get_keyterms_textrank(self) -> list:
        """
        Extract key terms using the TextRank algorithm.

        Returns:
            List[Tuple[str, float]]: A list of tuples containing key terms and their scores.
        """
        keyterms = extract.keyterms.textrank(
            self.text_doc, normalize="lemma", topn=self.top_n_values
        )
        return keyterms

    def get_keyterms_sgrank(self) -> list:
        """
        Extract key terms using the SGRank algorithm.

        Returns:
            List[Tuple[str, float]]: A list of tuples containing key terms and their scores.
        """
        keyterms = extract.keyterms.sgrank(
            self.text_doc, normalize="lemma", topn=self.top_n_values
        )
        return keyterms

    def get_keyterms_scake(self) -> list:
        """
        Extract key terms using the sCAKE algorithm.

        Returns:
            List[Tuple[str, float]]: A list of tuples containing key terms and their scores.
        """
        keyterms = extract.keyterms.scake(
            self.text_doc, normalize="lemma", topn=self.top_n_values
        )
        return keyterms

    def get_keyterms_yake(self) -> list:
        """
        Extract key terms using the YAKE algorithm.

        Returns:
            List[Tuple[str, float]]: A list of tuples containing key terms and their scores.
        """
        keyterms = extract.keyterms.yake(
            self.text_doc, normalize="lemma", topn=self.top_n_values
        )
        return keyterms

    def get_bigrams(self) -> list:
        """
        Extract bigrams from the text.

        Returns:
            List[str]: A list of bigram strings.
        """
        bigrams = extract.ngrams(
            self.text_doc,
            n=2,
            filter_stops=True,
            filter_nums=True,
            filter_punct=True,
            include_pos={"NOUN", "PROPN", "ADJ"},
        )
        return [ngram.text for ngram in bigrams]

    def get_trigrams(self) -> list:
        """
        Extract trigrams from the text.

        Returns:
            List[str]: A list of trigram strings.
        """
        trigrams = extract.ngrams(
            self.text_doc,
            n=3,
            filter_stops=True,
            filter_nums=True,
            filter_punct=True,
            include_pos={"NOUN", "PROPN", "ADJ"},
        )
        return [ngram.text for ngram in trigrams]

    def get_noun_chunks(self) -> list:
        """
        Extract noun chunks from the text.

        Returns:
            List[str]: A list of noun chunk strings.
        """
        noun_chunks = extract.noun_chunks(self.text_doc)
        return [chunk.text for chunk in noun_chunks]

    def get_entities(self) -> list:
        """
        Extract named entities from the text.

        Returns:
            List[str]: A list of named entity strings.
        """
        entities = extract.entities(self.text_doc)
        return [ent.text for ent in entities]

    def get_pos_tags(self) -> list:
        """
        Extract tokens and their part-of-speech tags.

        Returns:
            List[Tuple[str, str]]: A list of tuples containing tokens and their POS tags.
        """
        return [(token.text, token.pos_) for token in self.text_doc]
