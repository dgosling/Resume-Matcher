import re
from collections import defaultdict

import spacy
from spacy.matcher import PhraseMatcher

from .utils import (
    TextCleaner,
    REGEX_PATTERNS,
)  # Ensure REGEX_PATTERNS is imported if used

# Load the English model
nlp = spacy.load("en_core_web_sm")

RESUME_SECTIONS = [
    "Contact Information",
    "Objective",
    "Summary",
    "Education",
    "Experience",
    "Skills",
    "Projects",
    "Certifications",
    "Licenses",
    "Awards",
    "Honors",
    "Publications",
    "References",
    "Technical Skills",
    "Computer Skills",
    "Programming Languages",
    "Software Skills",
    "Soft Skills",
    "Language Skills",
    "Professional Skills",
    "Transferable Skills",
    "Work Experience",
    "Professional Experience",
    "Employment History",
    "Internship Experience",
    "Volunteer Experience",
    "Leadership Experience",
    "Research Experience",
    "Teaching Experience",
]


class DataExtractor:
    """
    A class for extracting various types of data from resume text.
    """

    def __init__(self, raw_text: str):
        """
        Initialize the DataExtractor object.

        Args:
            raw_text (str): The raw input text of the resume.
        """
        self.text = raw_text
        self.cleaner = TextCleaner()
        self.clean_text = self.cleaner.clean_text(self.text)
        self.doc = nlp(self.text)  # Use the original text for entity extraction
        self.clean_doc = nlp(
            self.clean_text
        )  # Use cleaned text for certain extractions
        self.nlp = nlp  # Assign the nlp model to self.nlp for easy access

    def extract_links(self) -> list:
        """
        Extract hyperlinks from the resume text.

        Returns:
            list: A list containing all the found hyperlinks.
        """
        link_pattern = r"(https?://\S+|www\.\S+)"
        links = re.findall(link_pattern, self.text)
        return links

    def extract_social_media_links(self) -> dict:
        """
        Extract social media profile links (LinkedIn, GitHub, etc.) from the resume text.

        Returns:
            dict: A dictionary with social media platform names as keys and URLs as values.
        """
        social_media_domains = {
            "linkedin.com": "LinkedIn",
            "github.com": "GitHub",
            "twitter.com": "Twitter",
            "facebook.com": "Facebook",
            "instagram.com": "Instagram",
            "stackoverflow.com": "StackOverflow",
            "medium.com": "Medium",
            "reddit.com": "Reddit",
            "behance.net": "Behance",
            "dribbble.com": "Dribbble",
        }
        links = self.extract_links()
        social_links = {}
        for link in links:
            for domain, platform in social_media_domains.items():
                if domain in link.lower():
                    social_links[platform] = link
        return social_links

    def extract_emails(self) -> list:
        """
        Extract email addresses from the resume text.

        Returns:
            list: A list containing all the extracted email addresses.
        """
        email_pattern = REGEX_PATTERNS["email_pattern"]
        emails = re.findall(email_pattern, self.text)
        return emails

    def extract_phone_numbers(self) -> list:
        """
        Extract phone numbers from the resume text.

        Returns:
            list: A list containing all the extracted phone numbers.
        """
        phone_pattern = r"""
            # Matches various phone number formats
            (?:(?:\+|00)\d{1,3}[-.\s]*)?     # Country code
            (?:\(?\d{1,4}\)?[-.\s]*)?        # Area code
            \d{1,4}[-.\s]*\d{1,4}[-.\s]*\d{1,9}   # Local number
        """
        phone_numbers = re.findall(phone_pattern, self.text, re.VERBOSE)
        phone_numbers = [number.strip() for number in phone_numbers if number.strip()]
        return phone_numbers

    def extract_names(self) -> list:
        """
        Extract names from the resume text using named entity recognition.

        Returns:
            list: A list of names extracted from the text.
        """
        names = [ent.text for ent in self.doc.ents if ent.label_ == "PERSON"]
        return names

    def extract_entities(self) -> list:
        """
        Extract organizations and locations from the resume text.

        Returns:
            list: A list of unique entities (organizations and locations).
        """
        entity_labels = ["ORG", "GPE", "LOC"]
        entities = [ent.text for ent in self.doc.ents if ent.label_ in entity_labels]
        return list(set(entities))

    def extract_skills(self, skills_list: list) -> list:
        """
        Extract skills from the resume text based on a predefined skills list using PhraseMatcher.

        Args:
            skills_list (list): A list of skills to search for in the text.

        Returns:
            list: A list of skills found in the resume text.
        """
        matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        patterns = [self.nlp.make_doc(skill.lower()) for skill in skills_list]
        matcher.add("Skills", patterns)
        matches = matcher(self.clean_doc)
        skills_found = [
            self.clean_doc[start:end].text for match_id, start, end in matches
        ]
        return list(set(skills_found))

    def extract_education(self) -> dict:
        """
        Extract education details from the resume text.

        Returns:
            dict: A dictionary containing degree, major, and institution information.
        """
        education = {}
        education_sections = self._extract_section("Education")
        if education_sections:
            education_text = " ".join(education_sections)
            doc = nlp(education_text)
            degrees = [
                "Bachelor",
                "Master",
                "B.Sc",
                "M.Sc",
                "PhD",
                "B.Tech",
                "M.Tech",
                "MBA",
                "B.E",
                "M.E",
                "BCA",
                "MCA",
            ]
            degree_pattern = re.compile(r"|".join(degrees), re.IGNORECASE)
            matches = degree_pattern.findall(education_text)
            if matches:
                education["Degree"] = matches[0]
            institutions = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
            if institutions:
                education["Institution"] = institutions[0]
        return education

    def extract_experience(self) -> list:
        """
        Extract work experience from the resume text.

        Returns:
            list: A list of experience descriptions extracted from the resume.
        """
        experiences = []
        experience_sections = self._extract_section("Experience")
        if experience_sections:
            experience_text = " ".join(experience_sections)
            experiences.append(experience_text.strip())
        return experiences

    def extract_sections(self) -> dict:
        """
        Split the resume text into sections based on common resume headings.

        Returns:
            dict: A dictionary with section titles as keys and their content as values.
        """
        section_titles = [re.escape(title) for title in RESUME_SECTIONS]
        section_pattern = r"(?P<section_title>^.*({0}).*$)".format(
            "|".join(section_titles)
        )
        regex = re.compile(section_pattern, re.IGNORECASE | re.MULTILINE)
        sections = defaultdict(list)
        lines = self.text.split("\n")
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            match = regex.match(line)
            if match:
                current_section = match.group("section_title").strip()
                sections[current_section] = []
                continue
            if current_section:
                sections[current_section].append(line)
        # Convert list of lines into a single string for each section
        sections = {k: "\n".join(v) for k, v in sections.items()}
        return sections

    def _extract_section(self, section_name: str) -> list:
        """
        Helper method to extract a specific section from the resume text.

        Args:
            section_name (str): The name of the section to extract.

        Returns:
            list: A list of lines belonging to the specified section.
        """
        sections = self.extract_sections()
        for key in sections.keys():
            if section_name.lower() in key.lower():
                return sections[key].split("\n")
        return []

    def extract_certifications(self) -> list:
        """
        Extract certifications and licenses from the resume text.

        Returns:
            list: A list of certifications found in the resume.
        """
        certification_sections = self._extract_section("Certifications")
        certifications = []
        if certification_sections:
            certifications = [
                line.strip() for line in certification_sections if line.strip()
            ]
        return certifications

    def extract_languages(self) -> list:
        """
        Extract languages known from the resume text.

        Returns:
            list: A list of languages mentioned in the resume.
        """
        languages = []
        language_sections = self._extract_section("Language Skills")
        if language_sections:
            language_text = " ".join(language_sections)
            doc = nlp(language_text)
            for ent in doc.ents:
                if ent.label_ == "LANGUAGE":
                    languages.append(ent.text)
        return list(set(languages))

    def extract_projects(self) -> list:
        """
        Extract project details from the resume text.

        Returns:
            list: A list of projects mentioned in the resume.
        """
        projects = []
        project_sections = self._extract_section("Projects")
        if project_sections:
            projects = [line.strip() for line in project_sections if line.strip()]
        return projects

    def extract_publications(self) -> list:
        """
        Extract publications from the resume text.

        Returns:
            list: A list of publications mentioned in the resume.
        """
        publications = []
        publication_sections = self._extract_section("Publications")
        if publication_sections:
            publications = [
                line.strip() for line in publication_sections if line.strip()
            ]
        return publications
