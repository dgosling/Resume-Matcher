import os
import re
from PyPDF2 import PdfReader


class DocumentLoader:
    @staticmethod
    def identify_file(file_path):
        _, extension = os.path.splitext(file_path)
        return extension.lower()

    @staticmethod
    def load_file(file_path):
        file_type = DocumentLoader.identify_file(file_path)
        if file_type == '.pdf':
            return PdfReader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_type}")


class TextExtractor:
    @staticmethod
    def extract_from_pdf(pdf_reader):
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    @staticmethod
    def clean_text(text):
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char.isspace())
        return text.strip()


def process_file(input_file, output_file):
    try:
        # Load the document
        document = DocumentLoader.load_file(input_file)

        # Extract and clean the text
        raw_text = TextExtractor.extract_from_pdf(document)
        cleaned_text = TextExtractor.clean_text(raw_text)

        print(cleaned_text)

        # Write the cleaned text to the output file
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(cleaned_text)

        print(f"Successfully processed {input_file} and saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


# Example usage
if __name__ == "__main__":
    input_file = 'alfred_pennyworth_pm.pdf'
    output_file = 'output.txt'
    process_file(input_file, output_file)