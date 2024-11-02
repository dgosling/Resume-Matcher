import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional

import spacy
import textacy
from sklearn.feature_extraction.text import TfidfVectorizer
from textacy import extract


@dataclass
class KeytermScore:
    """Data class to hold keyterm and its various scores"""

    term: str
    textrank_score: float = 0.0
    sgrank_score: float = 0.0
    scake_score: float = 0.0
    yake_score: float = 0.0
    tfidf_score: float = 0.0
    domain_relevance: float = 0.0
    combined_score: float = 0.0


class EnhancedKeytermExtractor:
    """
    Enhanced keyterm extraction with domain awareness and advanced scoring
    """

    # Common technical skills and domain-specific terms
    TECH_SKILLS = {
        "python",
        "java",
        "javascript",
        "react",
        "node.js",
        "sql",
        "aws",
        "docker",
        "kubernetes",
        "machine learning",
        "deep learning",
        "ai",
        "data science",
        "backend",
        "frontend",
        "full stack",
        "devops",
        "cloud computing",
        "microservices",
        "rest api",
        "graphql",
        "continuous integration",
        "continuous deployment",
        "agile",
    }

    # Common job titles and roles
    JOB_TITLES = {
        "software engineer",
        "developer",
        "architect",
        "data scientist",
        "product manager",
        "project manager",
        "team lead",
        "director",
        "vp",
        "chief",
        "specialist",
        "analyst",
        "consultant",
        "administrator",
    }

    def __init__(
        self,
        raw_text: str,
        top_n: int = 20,
        lang_model: str = "en_core_web_lg",
        domain_specific_terms: Optional[Set[str]] = None,
    ):
        """
        Initialize with enhanced options and domain knowledge

        Args:
            raw_text: Input text to process
            top_n: Number of top terms to return
            lang_model: SpaCy language model to use
            domain_specific_terms: Additional domain-specific terms to consider
        """
        self.raw_text = self._preprocess_text(raw_text)
        self.top_n = top_n
        self.nlp = spacy.load(lang_model)
        self.text_doc = textacy.make_spacy_doc(self.raw_text, lang=lang_model)

        # Combine default and custom domain terms
        self.domain_terms = self.TECH_SKILLS.union(self.JOB_TITLES)
        if domain_specific_terms:
            self.domain_terms.update(domain_specific_terms)

        # Initialize TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 3), stop_words="english", max_features=5000
        )

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r"\s+", " ", text)  # normalize whitespace
        text = re.sub(r"[^\w\s-]", " ", text)  # remove special chars except hyphens
        return text.lower().strip()

    def extract_all_keyterms(self) -> List[KeytermScore]:
        """Extract keyterms using multiple methods and combine scores"""
        # Get scores from different methods
        textrank_terms = dict(self.get_keyterms_based_on_textrank())
        sgrank_terms = dict(self.get_keyterms_based_on_sgrank())
        scake_terms = dict(self.get_keyterms_based_on_scake())
        yake_terms = dict(self.get_keyterms_based_on_yake())
        tfidf_terms = self._get_tfidf_scores()

        # Combine all unique terms
        all_terms = (
            set(textrank_terms)
            | set(sgrank_terms)
            | set(scake_terms)
            | set(yake_terms)
            | set(tfidf_terms)
        )

        # Create KeytermScore objects for each term
        keyterm_scores = []
        for term in all_terms:
            score = KeytermScore(
                term=term,
                textrank_score=textrank_terms.get(term, 0.0),
                sgrank_score=sgrank_terms.get(term, 0.0),
                scake_score=scake_terms.get(term, 0.0),
                yake_score=yake_terms.get(term, 0.0),
                tfidf_score=tfidf_terms.get(term, 0.0),
                domain_relevance=self._calculate_domain_relevance(term),
            )

            # Calculate combined score with weights
            score.combined_score = self._calculate_combined_score(score)
            keyterm_scores.append(score)

        # Sort by combined score and return top N
        return sorted(keyterm_scores, key=lambda x: x.combined_score, reverse=True)[
            : self.top_n
        ]

    def get_keyterms_based_on_textrank(self) -> List[Tuple[str, float]]:
        """Enhanced TextRank implementation"""
        return list(
            extract.keyterms.textrank(
                self.text_doc,
                normalize="lemma",
                topn=self.top_n,
                window_size=4,  # Increased window size for better context
                edge_weighting="count",  # Use count-based edge weighting
            )
        )

    def get_keyterms_based_on_sgrank(self) -> List[Tuple[str, float]]:
        """Enhanced SGRank implementation"""
        return list(
            extract.keyterms.sgrank(
                self.text_doc,
                normalize="lemma",
                topn=self.top_n,
                ngrams=(1, 2, 3),  # Consider up to trigrams
                windowsize=4,
                include_pos=("NOUN", "PROPN", "ADJ"),  # Include relevant POS tags
            )
        )

    def get_keyterms_based_on_scake(self) -> List[Tuple[str, float]]:
        """Enhanced sCAKE implementation"""
        return list(
            extract.keyterms.scake(
                self.text_doc,
                normalize="lemma",
                topn=self.top_n,
                include_pos=("NOUN", "PROPN", "ADJ"),
            )
        )

    def get_keyterms_based_on_yake(self) -> List[Tuple[str, float]]:
        """Enhanced YAKE implementation"""
        return list(
            extract.keyterms.yake(
                self.text_doc,
                normalize="lemma",
                topn=self.top_n,
                window_size=4,  # Increased window size
                include_pos=("NOUN", "PROPN", "ADJ", "VERB"),  # Include verbs
            )
        )

    def _get_tfidf_scores(self) -> Dict[str, float]:
        """Calculate TF-IDF scores for terms"""
        # Fit and transform the text
        tfidf_matrix = self.tfidf.fit_transform([self.raw_text])
        feature_names = self.tfidf.get_feature_names_out()

        # Get scores for each term
        scores = {}
        for i, score in enumerate(tfidf_matrix.toarray()[0]):
            if score > 0:  # Only include non-zero scores
                scores[feature_names[i]] = score

        return scores

    def _calculate_domain_relevance(self, term: str) -> float:
        """Calculate domain relevance score"""
        # Check exact matches
        if term in self.domain_terms:
            return 1.0

        # Check partial matches
        for domain_term in self.domain_terms:
            if domain_term in term or term in domain_term:
                return 0.75

        # Check if term contains common technical words
        technical_words = {"data", "software", "system", "api", "web", "cloud", "app"}
        if any(word in term for word in technical_words):
            return 0.5

        return 0.0

    def _calculate_combined_score(self, score: KeytermScore) -> float:
        """Calculate weighted combined score"""
        weights = {
            "textrank": 0.2,
            "sgrank": 0.2,
            "scake": 0.15,
            "yake": 0.15,
            "tfidf": 0.15,
            "domain": 0.15,
        }

        return (
            weights["textrank"] * score.textrank_score
            + weights["sgrank"] * score.sgrank_score
            + weights["scake"] * score.scake_score
            + weights["yake"] * score.yake_score
            + weights["tfidf"] * score.tfidf_score
            + weights["domain"] * score.domain_relevance
        )

    def get_ngrams(self, n: int = 2, min_freq: int = 2) -> List[Tuple[str, int]]:
        """Extract n-grams with minimum frequency requirement"""
        ngrams = textacy.extract.basics.ngrams(
            self.text_doc,
            n=n,
            filter_stops=True,
            filter_nums=True,
            filter_punct=True,
            min_freq=min_freq,
        )

        # Count frequencies
        ngram_freq = Counter(str(gram) for gram in ngrams)

        # Sort by frequency
        return sorted(ngram_freq.items(), key=lambda x: x[1], reverse=True)[
            : self.top_n
        ]

    def extract_key_phrases(self) -> List[str]:
        """Extract meaningful key phrases"""
        phrases = []

        # Extract noun phrases
        noun_phrases = textacy.extract.basics.noun_chunks(self.text_doc, min_freq=2)
        phrases.extend(str(phrase) for phrase in noun_phrases)

        # Extract verb phrases
        verb_phrases = textacy.extract.basics.verb_phrases(self.text_doc, min_freq=2)
        phrases.extend(str(phrase) for phrase in verb_phrases)

        # Remove duplicates and sort by length
        return sorted(set(phrases), key=len, reverse=True)[: self.top_n]

    def get_domain_specific_terms(self) -> List[Tuple[str, float]]:
        """Extract domain-specific terms with relevance scores"""
        terms_with_scores = []

        # Process text for domain-specific terms
        doc = self.nlp(self.raw_text)

        # Look for domain terms and variations
        for chunk in doc.noun_chunks:
            term = chunk.text.lower()
            score = self._calculate_domain_relevance(term)
            if score > 0:
                terms_with_scores.append((term, score))

        return sorted(terms_with_scores, key=lambda x: x[1], reverse=True)[: self.top_n]
