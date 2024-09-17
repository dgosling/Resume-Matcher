import glob
import os

from PyPDF2 import PdfReader


def get_pdf_files(directory_path: str) -> list:
    """
    Get a list of PDF files from the specified directory path.

    Args:
        directory_path (str): The directory path containing the PDF files.

    Returns:
        list: A list of PDF file paths.
    """
    if not os.path.isdir(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return []
    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
    return pdf_files


def read_single_pdf(file_path: str) -> str:
    """
    Read a single PDF file and extract the text from each page.

    Args:
        file_path (str): The path of the PDF file.

    Returns:
        str: A string containing the extracted text from the PDF file.
    """
    if not os.path.isfile(file_path):
        print(f"File '{file_path}' does not exist.")
        return ""
    output = []
    try:
        with open(file_path, "rb") as f:
            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    output.append(text)
    except Exception as e:
        print(f"Error reading file '{file_path}': {str(e)}")
    return " ".join(output)


def read_multiple_pdfs(directory_path: str) -> list:
    """
    Read multiple PDF files from the specified directory path and extract the text from each.

    Args:
        directory_path (str): The directory path containing the PDF files.

    Returns:
        list: A list containing the extracted text from each PDF file.
    """
    pdf_files = get_pdf_files(directory_path)
    all_texts = []
    for file_path in pdf_files:
        text = read_single_pdf(file_path)
        if text:
            all_texts.append(text)
    return all_texts
