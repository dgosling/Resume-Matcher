import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Set

import spacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span


@dataclass
class ContactInfo:
    emails: List[str]
    phones: List[str]
    links: List[str]
    location: Optional[str] = None


@dataclass
class Experience:
    title: str
    company: str
    start_date: str
    end_date: str
    description: str
    skills: List[str]


class EnhancedDataExtractor:
    """Enhanced data extraction using SpaCy with improved pattern matching and entity recognition."""

    # Improved phone regex patterns
    PHONE_PATTERNS = [
        r"\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",  # International
        r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",  # US/Canada
        r"\d{3}[-.\s]?\d{4}[-.\s]?\d{4}",  # Asian format
    ]

    # Improved email pattern
    EMAIL_PATTERN = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

    # Improved link patterns
    LINK_PATTERNS = [
        r"(?:https?:\/\/)?(?:www\.)?linkedin\.com\/in\/[\w\-]+\/?",
        r"(?:https?:\/\/)?(?:www\.)?github\.com\/[\w\-]+\/?",
        r"(?:https?:\/\/)?(?:[\w-])+\.(?:[\w-])+(?:\/[\w-]+)*\/?",
    ]

    def __init__(self, model_name: str = "en_core_web_lg"):
        """Initialize with a larger SpaCy model for better accuracy."""
        self.nlp = spacy.load(model_name)
        self._setup_matchers()
        self._setup_custom_pipes()

    def _setup_custom_pipes(self):
        """Add custom pipeline components to SpaCy."""
        # Add custom components here if needed
        if "section_detector" not in self.nlp.pipe_names:
            self.nlp.add_pipe("section_detector", before="ner")

    def _setup_matchers(self):
        """Setup SpaCy matchers for various patterns."""
        self.matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab)

        # Add section header patterns
        section_patterns = [
            [
                {"LOWER": {"IN": ["experience", "work", "employment"]}},
                {"LOWER": "history", "OP": "?"},
            ],
            [
                {"LOWER": {"IN": ["education", "academic"]}},
                {"LOWER": "background", "OP": "?"},
            ],
            [
                {"LOWER": {"IN": ["skills", "technologies", "technical"]}},
                {"LOWER": "competencies", "OP": "?"},
            ],
        ]
        self.matcher.add("SECTION_HEADERS", section_patterns)

        # Add job title patterns
        job_patterns = [
            [{"POS": "PROPN"}, {"LOWER": "engineer"}],
            [{"LOWER": "senior"}, {"POS": "PROPN"}, {"LOWER": "engineer"}],
            [{"POS": "PROPN"}, {"LOWER": "developer"}],
            # Add more job title patterns
        ]
        self.matcher.add("JOB_TITLES", job_patterns)

    def extract_all(self, text: str) -> Dict:
        """Extract all relevant information from the text."""
        doc = self.nlp(text)

        return {
            "contact_info": self.extract_contact_info(doc),
            "experiences": self.extract_experiences(doc),
            "education": self.extract_education(doc),
            "skills": self.extract_skills(doc),
            "entities": self.extract_entities(doc),
        }

    def extract_contact_info(self, doc: Doc) -> ContactInfo:
        """Extract contact information using improved patterns."""
        emails = set(re.findall(self.EMAIL_PATTERN, doc.text))
        phones = set()
        for pattern in self.PHONE_PATTERNS:
            phones.update(re.findall(pattern, doc.text))

        links = set()
        for pattern in self.LINK_PATTERNS:
            links.update(re.findall(pattern, doc.text))

        # Extract location using NER
        locations = [ent.text for ent in doc.ents if ent.label_ in {"GPE", "LOC"}]
        location = locations[0] if locations else None

        return ContactInfo(
            emails=list(emails),
            phones=list(phones),
            links=list(links),
            location=location,
        )

    def extract_experiences(self, doc: Doc) -> List[Experience]:
        """Extract work experiences with improved accuracy."""
        experiences = []
        experience_sections = self._get_sections(doc, "experience")

        for section in experience_sections:
            # Use custom rules and NER to extract experience details
            experiences.extend(self._parse_experience_section(section))

        return experiences

    def _parse_experience_section(self, section: Span) -> List[Experience]:
        """Parse individual experience sections with improved accuracy."""
        experiences = []
        current_exp = None

        for sent in section.sents:
            # Look for job titles
            title_matches = self.matcher(sent, type="JOB_TITLES")
            if title_matches:
                if current_exp:
                    experiences.append(current_exp)

                # Extract company using NER
                companies = [ent.text for ent in sent.ents if ent.label_ == "ORG"]
                company = companies[0] if companies else ""

                # Extract dates
                dates = self._extract_dates(sent)
                start_date = dates[0] if dates else ""
                end_date = dates[-1] if len(dates) > 1 else "Present"

                current_exp = Experience(
                    title=sent[title_matches[0][1] : title_matches[0][2]].text,
                    company=company,
                    start_date=start_date,
                    end_date=end_date,
                    description="",
                    skills=[],
                )
            elif current_exp:
                # Add to description and extract skills
                current_exp.description += sent.text + " "
                current_exp.skills.extend(self._extract_skills_from_text(sent))

        if current_exp:
            experiences.append(current_exp)

        return experiences

    def _extract_dates(self, span: Span) -> List[str]:
        """Extract dates with improved pattern matching."""
        date_patterns = [
            r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
            r"Dec(?:ember)?)[,\s]+\d{4}",
            r"\d{4}",
        ]

        dates = []
        for pattern in date_patterns:
            matches = re.finditer(pattern, span.text)
            dates.extend(match.group() for match in matches)

        return sorted(dates)

    def extract_skills(self, doc: Doc) -> Set[str]:
        """Extract skills using improved detection methods."""
        skills = set()

        # Use custom skill patterns and known skill lists
        skill_sections = self._get_sections(doc, "skills")
        for section in skill_sections:
            # Extract individual skills
            for token in section:
                if token.pos_ in {"PROPN", "NOUN"} and len(token.text) > 2:
                    skills.add(token.text)

        # Add skills found in experience sections
        for exp_section in self._get_sections(doc, "experience"):
            skills.update(self._extract_skills_from_text(exp_section))

        return skills

    def _extract_skills_from_text(self, span: Span) -> Set[str]:
        """Extract skills from a text span using improved detection."""
        skills = set()

        # Add your custom skill detection logic here
        # This could include:
        # 1. Known skill dictionary matching
        # 2. Technical term extraction
        # 3. Framework and tool name detection

        return skills

    def _get_sections(self, doc: Doc, section_type: str) -> List[Span]:
        """Get document sections with improved section detection."""
        sections = []
        current_section = None
        current_text = []

        for sent in doc.sents:
            # Check if this sentence is a section header
            matches = self.matcher(sent)
            is_header = any(
                self.nlp.vocab.strings[match_id] == "SECTION_HEADERS"
                for match_id, _, _ in matches
            )

            if is_header:
                if current_section and current_text:
                    sections.append(doc[current_section : sent.start])
                current_section = sent.start
                current_text = []
            elif current_section is not None:
                current_text.append(sent.text)

        # Add the last section
        if current_section and current_text:
            sections.append(doc[current_section:])

        return sections

    def extract_entities(self, doc: Doc) -> Dict[str, List[str]]:
        """Extract named entities with improved categorization."""
        entities = {
            "organizations": [],
            "locations": [],
            "dates": [],
            "skills": [],
            "names": [],
        }

        for ent in doc.ents:
            if ent.label_ == "ORG":
                entities["organizations"].append(ent.text)
            elif ent.label_ in {"GPE", "LOC"}:
                entities["locations"].append(ent.text)
            elif ent.label_ == "DATE":
                entities["dates"].append(ent.text)
            elif ent.label_ == "PERSON":
                entities["names"].append(ent.text)

        return entities

    def extract_education(self, doc):
        pass
