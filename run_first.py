import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from scripts import JobDescriptionProcessor, ResumeProcessor
from scripts.utils import init_logging_config


class DocumentType(Enum):
    RESUME = "resume"
    JOB_DESCRIPTION = "job_description"


@dataclass
class ProcessingConfig:
    base_dir: Path = Path("Data")
    input_resume_dir: Path = Path("Data/Resumes")
    input_job_dir: Path = Path("Data/JobDescription")
    processed_resume_dir: Path = Path("Data/Processed/Resumes")
    processed_job_dir: Path = Path("Data/Processed/JobDescription")


class DocumentProcessor:
    """Main class to handle document processing for both resumes and job descriptions."""

    def __init__(self, config: ProcessingConfig = ProcessingConfig()):
        self.config = config
        init_logging_config()
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Ensure all necessary directories exist."""
        directories = [
            self.config.processed_resume_dir,
            self.config.processed_job_dir,
            self.config.input_resume_dir,
            self.config.input_job_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _clean_directory(self, directory: Path) -> None:
        """Remove all files from the specified directory."""
        try:
            for file_path in directory.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
            logging.info(f"Cleaned directory: {directory}")
        except Exception as e:
            logging.error(f"Error cleaning directory {directory}: {e}")

    def _process_documents(
        self, doc_type: DocumentType, input_dir: Path, processed_dir: Path
    ) -> bool:
        """Process all documents of a specific type."""
        try:
            self._clean_directory(processed_dir)

            files = list(input_dir.glob("*"))
            if not files:
                logging.error(f"No {doc_type.value}s found in {input_dir}")
                return False

            logging.info(f"Processing {len(files)} {doc_type.value}s")

            for file_path in files:
                processor_class = (
                    ResumeProcessor
                    if doc_type == DocumentType.RESUME
                    else JobDescriptionProcessor
                )
                processor = processor_class(str(file_path))
                if not processor.process():
                    logging.warning(f"Failed to process {file_path}")

            logging.info(f"Completed processing {doc_type.value}s")
            return True

        except Exception as e:
            logging.error(f"Error processing {doc_type.value}s: {e}")
            return False

    def process_all(self) -> bool:
        """Process both resumes and job descriptions."""
        resume_success = self._process_documents(
            DocumentType.RESUME,
            self.config.input_resume_dir,
            self.config.processed_resume_dir,
        )

        job_desc_success = self._process_documents(
            DocumentType.JOB_DESCRIPTION,
            self.config.input_job_dir,
            self.config.processed_job_dir,
        )

        if resume_success and job_desc_success:
            logging.info("Successfully processed all documents")
            logging.info("You can now run 'streamlit run streamlit_second.py'")
            return True
        return False


def main():
    """Main entry point for the document processing script."""
    processor = DocumentProcessor()
    if not processor.process_all():
        exit(1)


if __name__ == "__main__":
    main()
