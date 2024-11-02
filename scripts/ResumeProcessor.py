import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

from .ReadPdf import EnhancedPDFProcessor
from .parsers import ParseJobDesc, ParseResume

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration settings for resume processing."""

    input_directory: Path = Path("Data/Resumes/")
    output_directory: Path = Path("Data/Processed/Resumes")
    backup_directory: Path = Path("Data/Backup/Resumes")
    file_prefix: str = "Resume-"
    indent_level: int = 4
    create_backup: bool = True
    validate_pdf: bool = True
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    max_workers: int = 4


class ResumeProcessor:
    """A class for processing resume PDFs and job descriptions.

    Features:
    - PDF validation
    - Automatic directory creation
    - Backup creation
    - Detailed logging
    - Error handling
    - Batch processing
    """

    def __init__(
        self, input_file: Union[str, Path], config: Optional[ProcessingConfig] = None
    ):
        """Initialize the ResumeProcessor.

        Args:
            input_file: Path to the input PDF file
            config: Optional processing configuration

        Raises:
            ValueError: If input file is invalid
        """
        self.config = config or ProcessingConfig()
        self.input_file = Path(input_file)
        self.input_file_name = self.config.input_directory / self.input_file

        # Initialize PDF processor
        self.pdf_processor = EnhancedPDFProcessor(
            max_workers=self.config.max_workers,
            max_file_size=self.config.max_file_size,
        )

        # Initialize processing metadata
        self.processed_at = None
        self.processing_duration = None
        self.error_details = None

        # Ensure directories exist
        self._create_directories()

    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.config.input_directory,
            self.config.output_directory,
            self.config.backup_directory,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _validate_pdf(self) -> bool:
        """Validate if the input file is a valid PDF.

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Using EnhancedPDFProcessor's validation logic
            valid_files = self.pdf_processor.get_pdf_files(self.input_file_name.parent)
            return self.input_file_name in valid_files
        except Exception as e:
            logger.error(f"PDF validation failed: {str(e)}")
            return False

    def _create_backup(self, data: Dict) -> None:
        """Create a backup of the processed data.

        Args:
            data: Dictionary containing processed resume data
        """
        if not self.config.create_backup:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = (
            self.config.backup_directory
            / f"backup_{self.input_file.stem}_{timestamp}.json"
        )

        try:
            self._write_json_to_file(backup_file, data)
            logger.info(f"Backup created: {backup_file}")
        except Exception as e:
            logger.warning(f"Failed to create backup: {str(e)}")

    def _write_json_to_file(self, file_path: Path, data: Dict) -> None:
        """Write JSON data to file with error handling.

        Args:
            file_path: Path to output file
            data: Dictionary to be written

        Raises:
            IOError: If writing fails
        """
        try:
            json_object = json.dumps(
                data,
                sort_keys=True,
                indent=self.config.indent_level,
                ensure_ascii=False,
            )

            with open(file_path, "w+", encoding="utf-8") as outfile:
                outfile.write(json_object)

        except Exception as e:
            logger.error(f"Failed to write JSON file: {str(e)}")
            raise IOError(f"Failed to write file {file_path}: {str(e)}")

    def _read_resumes(self) -> Dict:
        """Read and parse resume PDF.

        Returns:
            Dict: Parsed resume data

        Raises:
            ValueError: If PDF reading or parsing fails
        """
        try:
            # Using new EnhancedPDFProcessor to read the PDF
            pdf_data = self.pdf_processor.read_pdf(self.input_file_name)

            # Extract full text from the PDF data
            text_content = pdf_data["full_text"]

            # Parse the extracted text
            output = ParseResume(text_content).get_JSON()

            # Add metadata from EnhancedPDFProcessor
            output["processing_metadata"] = {
                "processed_at": datetime.now().isoformat(),
                "processor_version": "3.0.0",  # Updated version
                "original_filename": str(self.input_file),
                "pdf_metadata": pdf_data["metadata"].__dict__,  # Include PDF metadata
            }

            return output

        except Exception as e:
            logger.error(f"Failed to read resume: {str(e)}")
            raise ValueError(f"Resume parsing failed: {str(e)}")

    def _read_job_desc(self) -> Dict:
        """Read and parse job description PDF.

        Returns:
            Dict: Parsed job description data

        Raises:
            ValueError: If PDF reading or parsing fails
        """
        try:
            # Using new EnhancedPDFProcessor to read the PDF
            pdf_data = self.pdf_processor.read_pdf(self.input_file_name)

            # Extract full text from the PDF data
            text_content = pdf_data["full_text"]

            # Parse the extracted text
            output = ParseJobDesc(text_content).get_JSON()

            # Add metadata from EnhancedPDFProcessor
            output["processing_metadata"] = {
                "processed_at": datetime.now().isoformat(),
                "processor_version": "3.0.0",  # Updated version
                "original_filename": str(self.input_file),
                "pdf_metadata": pdf_data["metadata"].__dict__,  # Include PDF metadata
            }

            return output

        except Exception as e:
            logger.error(f"Failed to read job description: {str(e)}")
            raise ValueError(f"Job description parsing failed: {str(e)}")

    def _write_json_file(self, resume_dictionary: Dict) -> None:
        """Write processed resume data to JSON file.

        Args:
            resume_dictionary: Processed resume data
        """
        file_name = (
            f"{self.config.file_prefix}"
            f"{self.input_file.stem}_"
            f"{resume_dictionary['unique_id']}.json"
        )

        save_path = self.config.output_directory / file_name
        self._write_json_to_file(save_path, resume_dictionary)
        logger.info(f"Successfully wrote output to {save_path}")

    def process(self) -> bool:
        """Process the resume file.

        Returns:
            bool: True if processing successful, False otherwise
        """
        start_time = datetime.now()

        try:
            # Validate PDF if enabled
            if self.config.validate_pdf and not self._validate_pdf():
                return False

            # Process resume
            resume_dict = self._read_resumes()

            # Create backup if enabled
            self._create_backup(resume_dict)

            # Write output file
            self._write_json_file(resume_dict)

            # Update processing metadata
            self.processed_at = datetime.now()
            self.processing_duration = (datetime.now() - start_time).total_seconds()

            logger.info(f"Successfully processed {self.input_file}")
            return True

        except Exception as e:
            self.error_details = str(e)
            logger.error(f"Processing failed: {str(e)}")
            return False

    @classmethod
    def batch_process(
        cls,
        input_directory: Union[str, Path],
        config: Optional[ProcessingConfig] = None,
    ) -> Dict[str, bool]:
        """Process multiple resume files in a directory.

        Args:
            input_directory: Directory containing resume PDFs
            config: Optional processing configuration

        Returns:
            Dict mapping filenames to processing success status
        """
        input_path = Path(input_directory)
        results = {}

        if not input_path.exists():
            logger.error(f"Input directory not found: {input_path}")
            return results

        # Initialize PDF processor for batch processing
        pdf_processor = EnhancedPDFProcessor(
            max_workers=config.max_workers if config else ProcessingConfig().max_workers
        )

        # Get valid PDF files using EnhancedPDFProcessor
        try:
            pdf_files = pdf_processor.get_pdf_files(input_path)
            for pdf_file in pdf_files:
                processor = cls(pdf_file.name, config)
                results[pdf_file.name] = processor.process()
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")

        return results

    def get_processing_status(self) -> Dict:
        """Get the current processing status and metadata.

        Returns:
            Dict containing processing status information
        """
        return {
            "file_name": str(self.input_file),
            "processed_at": (
                self.processed_at.isoformat() if self.processed_at else None
            ),
            "processing_duration": self.processing_duration,
            "error_details": self.error_details,
            "success": bool(self.processed_at and not self.error_details),
        }
