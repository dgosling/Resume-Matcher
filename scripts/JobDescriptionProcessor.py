import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union, List

from .ReadPdf import EnhancedPDFProcessor
from .parsers import ParseJobDesc

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class JobDescriptionConfig:
    """Configuration settings for job description processing."""

    input_directory: Path = Path("Data/JobDescription/")
    output_directory: Path = Path("Data/Processed/JobDescription")
    backup_directory: Path = Path("Data/Backup/JobDescription")
    file_prefix: str = "JobDescription-"
    indent_level: int = 4
    create_backup: bool = True
    validate_pdf: bool = True
    extract_skills: bool = True
    extract_requirements: bool = True
    extract_qualifications: bool = True
    max_pdf_size: int = 100 * 1024 * 1024  # 100MB
    pdf_chunk_size: int = 1024 * 1024  # 1MB
    max_workers: int = 4


class JobDescriptionProcessor:
    """A class for processing job description PDFs.

    Features:
    - Enhanced PDF validation and processing
    - Automatic directory creation
    - Backup creation
    - Detailed logging
    - Error handling
    - Batch processing
    - Skills extraction
    - Requirements analysis
    - PDF metadata extraction
    """

    def __init__(
        self,
        input_file: Union[str, Path],
        config: Optional[JobDescriptionConfig] = None,
    ):
        """Initialize the JobDescriptionProcessor.

        Args:
            input_file: Path to the input PDF file
            config: Optional processing configuration

        Raises:
            ValueError: If input file is invalid
        """
        self.config = config or JobDescriptionConfig()
        self.input_file = Path(input_file)
        self.input_file_name = self.config.input_directory / self.input_file

        # Initialize PDF processor
        self.pdf_processor = EnhancedPDFProcessor(
            max_workers=self.config.max_workers,
            chunk_size=self.config.pdf_chunk_size,
            max_file_size=self.config.max_pdf_size,
        )

        # Initialize processing metadata
        self.processed_at = None
        self.processing_duration = None
        self.error_details = None
        self.extracted_metadata = {}
        self.pdf_metadata = None

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
        """Validate if the input file is a valid PDF using enhanced validation.

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Use enhanced PDF processor's validation
            valid_files = self.pdf_processor.get_pdf_files(self.input_file_name.parent)
            return self.input_file_name in valid_files
        except Exception as e:
            logger.error(f"PDF validation failed: {str(e)}")
            return False

    def _read_job_desc(self) -> Dict:
        """Read and parse job description PDF using enhanced PDF processor.

        Returns:
            Dict: Parsed job description data

        Raises:
            ValueError: If PDF reading or parsing fails
        """
        try:
            # Use enhanced PDF processor to read the file
            pdf_data = self.pdf_processor.read_pdf(self.input_file_name)

            # Store PDF metadata
            self.pdf_metadata = pdf_data["metadata"]

            # Get full text content
            content = pdf_data["full_text"]

            # Parse the content
            output = ParseJobDesc(content).get_JSON()

            # Add metadata
            output["processing_metadata"] = self._extract_job_metadata(output, pdf_data)

            # Store extracted metadata for later use
            self.extracted_metadata = output["processing_metadata"]

            return output

        except Exception as e:
            logger.error(f"Failed to read job description: {str(e)}")
            raise ValueError(f"Job description parsing failed: {str(e)}")

    def _extract_job_metadata(self, job_data: Dict, pdf_data: Dict) -> Dict:
        """Extract additional metadata from job description.

        Args:
            job_data: Parsed job description data
            pdf_data: Raw PDF data from enhanced processor

        Returns:
            Dict containing extracted metadata
        """
        metadata = {
            "processing_timestamp": datetime.now().isoformat(),
            "source_file": str(self.input_file),
            "pdf_metadata": {
                "num_pages": pdf_data["metadata"].num_pages,
                "file_size": pdf_data["metadata"].file_size,
                "author": pdf_data["metadata"].author,
                "title": pdf_data["metadata"].title,
                "creation_date": pdf_data["metadata"].creation_date,
                "checksum": pdf_data["metadata"].checksum,
            },
            "extracted_skills": [],
            "extracted_requirements": [],
            "extracted_qualifications": [],
        }

        try:
            # Analyze PDF structure
            structure_analysis = self.pdf_processor.analyze_pdf_structure(
                self.input_file_name
            )
            metadata["pdf_structure"] = structure_analysis
        except Exception as e:
            logger.warning(f"Failed to analyze PDF structure: {str(e)}")

        # Add extraction logic based on config
        if self.config.extract_skills:
            metadata["extracted_skills"] = self._extract_skills(pdf_data["full_text"])

        if self.config.extract_requirements:
            metadata["extracted_requirements"] = self._extract_requirements(
                pdf_data["full_text"]
            )

        if self.config.extract_qualifications:
            metadata["extracted_qualifications"] = self._extract_qualifications(
                pdf_data["full_text"]
            )

        return metadata

    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from job description text.

        Args:
            text: Full text content

        Returns:
            List of extracted skills
        """
        # TODO: Implement skill extraction logic
        return []

    def _extract_requirements(self, text: str) -> List[str]:
        """Extract requirements from job description text.

        Args:
            text: Full text content

        Returns:
            List of extracted requirements
        """
        # TODO: Implement requirements extraction logic
        return []

    def _extract_qualifications(self, text: str) -> List[str]:
        """Extract qualifications from job description text.

        Args:
            text: Full text content

        Returns:
            List of extracted qualifications
        """
        # TODO: Implement qualifications extraction logic
        return []

    def process(self) -> bool:
        """Process the job description file.

        Returns:
            bool: True if processing successful, False otherwise
        """
        start_time = datetime.now()

        try:
            # Validate PDF if enabled
            if self.config.validate_pdf and not self._validate_pdf():
                return False

            # Process job description
            job_desc_dict = self._read_job_desc()

            # Create backup if enabled
            self._create_backup(job_desc_dict)

            # Write output file
            self._write_json_file(job_desc_dict)

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
        config: Optional[JobDescriptionConfig] = None,
    ) -> Dict[str, bool]:
        """Process multiple job description files in a directory using parallel processing.

        Args:
            input_directory: Directory containing job description PDFs
            config: Optional processing configuration

        Returns:
            Dict mapping filenames to processing success status
        """
        input_path = Path(input_directory)
        results = {}

        if not input_path.exists():
            logger.error(f"Input directory not found: {input_path}")
            return results

        # Use enhanced PDF processor for batch processing
        processor = cls("dummy.pdf", config)  # Temporary instance for PDF processor

        try:
            # Get valid PDF files
            pdf_files = processor.pdf_processor.get_pdf_files(input_path)

            # Process each valid PDF
            for pdf_file in pdf_files:
                job_processor = cls(pdf_file.name, config)
                results[pdf_file.name] = job_processor.process()

        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")

        return results

    def get_processing_status(self) -> Dict:
        """Get the current processing status and metadata.

        Returns:
            Dict containing processing status, metadata, and PDF information
        """
        return {
            "file_name": str(self.input_file),
            "processed_at": (
                self.processed_at.isoformat() if self.processed_at else None
            ),
            "processing_duration": self.processing_duration,
            "error_details": self.error_details,
            "success": bool(self.processed_at and not self.error_details),
            "extracted_metadata": self.extracted_metadata,
            "pdf_metadata": self.pdf_metadata._asdict() if self.pdf_metadata else None,
        }

    def get_extracted_data(self) -> Dict:
        """Get the extracted metadata from the job description.

        Returns:
            Dict containing extracted skills, requirements, qualifications, and PDF metadata
        """
        return self.extracted_metadata or {}
