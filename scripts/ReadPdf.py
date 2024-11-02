import hashlib
import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Generator, Union

from pypdf import PdfReader


@dataclass
class PDFMetadata:
    """Metadata for a PDF file"""

    filename: str
    num_pages: int
    file_size: int
    author: Optional[str] = None
    title: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    creation_date: Optional[str] = None
    checksum: Optional[str] = None


@dataclass
class PDFPage:
    """Representation of a single PDF page"""

    page_number: int
    content: str
    file_name: str
    tables: List[List[str]] = None
    images: List[Dict] = None


class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors"""

    pass


class EnhancedPDFProcessor:
    """Enhanced PDF processing with better error handling and features"""

    def __init__(
        self,
        max_workers: int = 4,
        chunk_size: int = 1024 * 1024,  # 1MB
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        logging_level: int = logging.INFO,
    ):
        """
        Initialize the PDF processor

        Args:
            max_workers: Maximum number of worker threads
            chunk_size: Size of chunks for reading large files
            max_file_size: Maximum allowed file size
            logging_level: Logging level
        """
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.max_file_size = max_file_size

        # Set up logging
        logging.basicConfig(
            level=logging_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def get_pdf_files(self, directory: Union[str, Path]) -> List[Path]:
        """
        Get all PDF files from a directory with validation

        Args:
            directory: Directory path to search for PDFs

        Returns:
            List of Path objects for valid PDF files

        Raises:
            PDFProcessingError: If directory doesn't exist or other errors
        """
        try:
            directory = Path(directory)
            if not directory.exists():
                raise PDFProcessingError(f"Directory does not exist: {directory}")
            if not directory.is_dir():
                raise PDFProcessingError(f"Path is not a directory: {directory}")

            pdf_files = list(directory.glob("**/*.pdf"))  # Recursive search

            # Validate each PDF file
            valid_files = []
            for pdf_file in pdf_files:
                try:
                    if pdf_file.stat().st_size > self.max_file_size:
                        self.logger.warning(
                            f"Skipping file exceeding size limit: {pdf_file}"
                        )
                        continue

                    # Basic PDF header check
                    with open(pdf_file, "rb") as f:
                        header = f.read(5)
                        if header.startswith(b"%PDF-"):
                            valid_files.append(pdf_file)
                        else:
                            self.logger.warning(
                                f"Invalid PDF header in file: {pdf_file}"
                            )
                except Exception as e:
                    self.logger.error(f"Error validating {pdf_file}: {str(e)}")

            return valid_files

        except Exception as e:
            raise PDFProcessingError(f"Error getting PDF files: {str(e)}")

    def read_pdf(self, pdf_path: Union[str, Path]) -> Dict:
        """
        Read a single PDF file with enhanced features

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary containing PDF content and metadata

        Raises:
            PDFProcessingError: If reading fails
        """
        try:
            pdf_path = Path(pdf_path)

            if not pdf_path.exists():
                raise PDFProcessingError(f"File does not exist: {pdf_path}")

            # Calculate file checksum
            checksum = self._calculate_checksum(pdf_path)

            with open(pdf_path, "rb") as f:
                reader = PdfReader(f)

                # Extract metadata
                metadata = self._extract_metadata(reader, pdf_path, checksum)

                # Extract content with parallel processing
                pages = self._extract_pages_parallel(reader, str(pdf_path))

                return {
                    "metadata": metadata,
                    "pages": pages,
                    "full_text": self._combine_page_contents(pages),
                }

        except Exception as e:
            raise PDFProcessingError(f"Error reading PDF {pdf_path}: {str(e)}")

    def read_multiple_pdfs(
        self, directory: Union[str, Path], parallel: bool = True
    ) -> Generator[Dict, None, None]:
        """
        Read multiple PDFs with parallel processing option

        Args:
            directory: Directory containing PDF files
            parallel: Whether to use parallel processing

        Yields:
            Dictionary for each PDF with content and metadata
        """
        pdf_files = self.get_pdf_files(directory)

        if parallel and len(pdf_files) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_pdf = {
                    executor.submit(self.read_pdf, pdf_file): pdf_file
                    for pdf_file in pdf_files
                }

                for future in as_completed(future_to_pdf):
                    pdf_file = future_to_pdf[future]
                    try:
                        result = future.result()
                        yield result
                    except Exception as e:
                        self.logger.error(f"Error processing {pdf_file}: {str(e)}")
        else:
            for pdf_file in pdf_files:
                try:
                    yield self.read_pdf(pdf_file)
                except Exception as e:
                    self.logger.error(f"Error processing {pdf_file}: {str(e)}")

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of file"""
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(self.chunk_size), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def _extract_metadata(
        self, reader: PdfReader, file_path: Path, checksum: str
    ) -> PDFMetadata:
        """Extract metadata from PDF"""
        try:
            info = reader.metadata
            return PDFMetadata(
                filename=file_path.name,
                num_pages=len(reader.pages),
                file_size=file_path.stat().st_size,
                author=info.get("/Author", None),
                title=info.get("/Title", None),
                subject=info.get("/Subject", None),
                creator=info.get("/Creator", None),
                creation_date=info.get("/CreationDate", None),
                checksum=checksum,
            )
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {str(e)}")
            return PDFMetadata(
                filename=file_path.name,
                num_pages=len(reader.pages),
                file_size=file_path.stat().st_size,
                checksum=checksum,
            )

    def _extract_pages_parallel(
        self, reader: PdfReader, file_name: str
    ) -> List[PDFPage]:
        """Extract pages in parallel"""
        pages = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_page = {
                executor.submit(
                    self._extract_single_page, reader, page_num, file_name
                ): page_num
                for page_num in range(len(reader.pages))
            }

            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    page = future.result()
                    pages.append(page)
                except Exception as e:
                    self.logger.error(f"Error extracting page {page_num}: {str(e)}")

        return sorted(pages, key=lambda x: x.page_number)

    def _extract_single_page(
        self, reader: PdfReader, page_num: int, file_name: str
    ) -> PDFPage:
        """Extract content from a single page"""
        page = reader.pages[page_num]
        content = page.extract_text()

        # Clean and normalize text
        content = self._clean_text(content)

        return PDFPage(page_number=page_num + 1, content=content, file_name=file_name)

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Fix common OCR errors
        text = text.replace("|", "I")
        text = text.replace("0", "O")

        # Remove non-printable characters
        text = "".join(char for char in text if char.isprintable())

        return text.strip()

    @staticmethod
    def _combine_page_contents(pages: List[PDFPage]) -> str:
        """Combine contents of all pages"""
        return "\n\n".join(page.content for page in pages)

    def analyze_pdf_structure(self, pdf_path: Union[str, Path]) -> Dict:
        """
        Analyze the structure of a PDF

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary containing structural analysis
        """
        try:
            pdf_path = Path(pdf_path)
            with open(pdf_path, "rb") as f:
                reader = PdfReader(f)

                # Analyze structure
                structure = {
                    "num_pages": len(reader.pages),
                    "has_text": defaultdict(bool),
                    "has_images": defaultdict(bool),
                    "has_tables": defaultdict(bool),
                    "font_info": defaultdict(set),
                }

                # Analyze each page
                for page_num, page in enumerate(reader.pages, 1):
                    # Check for text
                    text = page.extract_text()
                    structure["has_text"][page_num] = bool(text.strip())

                    # TODO: Implement image and table detection
                    # This would require additional libraries like
                    # pdf2image and tabula-py

                return structure

        except Exception as e:
            raise PDFProcessingError(f"Error analyzing PDF structure: {str(e)}")


def main():
    """Example usage"""
    processor = EnhancedPDFProcessor()

    # Process single PDF
    try:
        result = processor.read_pdf("example.pdf")
        print(f"Metadata: {result['metadata']}")
        print(f"Total pages: {len(result['pages'])}")
    except PDFProcessingError as e:
        logging.error(f"Error processing PDF: {str(e)}")

    # Process multiple PDFs
    try:
        for pdf_result in processor.read_multiple_pdfs("pdf_directory"):
            print(f"Processed: {pdf_result['metadata'].filename}")
    except PDFProcessingError as e:
        logging.error(f"Error processing PDFs: {str(e)}")


if __name__ == "__main__":
    main()
