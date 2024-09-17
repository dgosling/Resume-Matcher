import json
import logging
import os
import sys

from scripts import JobDescriptionProcessor, ResumeProcessor
from scripts.utils import get_filenames_from_dir, init_logging_config

init_logging_config()

PROCESSED_RESUMES_PATH = "Data/Processed/Resumes"
PROCESSED_JOB_DESCRIPTIONS_PATH = "Data/Processed/JobDescription"


def read_json(filename):
    """
    Read a JSON file and return its contents.

    Args:
        filename (str): The path to the JSON file.

    Returns:
        dict: The data contained in the JSON file.
    """
    with open(filename) as f:
        data = json.load(f)
    return data


def remove_old_files(files_path):
    """
    Remove all files from the specified directory.

    Args:
        files_path (str): The directory path from which to remove files.
    """
    if not os.path.exists(files_path):
        logging.warning(f"Directory '{files_path}' does not exist. Creating it.")
        os.makedirs(files_path)
    for filename in os.listdir(files_path):
        try:
            file_path = os.path.join(files_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            logging.error(f"Error deleting '{file_path}': {e}")
    logging.info(f"Deleted old files from '{files_path}'.")


logging.info("Started to read from 'Data/Resumes'")
try:
    remove_old_files(PROCESSED_RESUMES_PATH)
    file_names = get_filenames_from_dir("Data/Resumes")
    if not file_names:
        raise FileNotFoundError("No resume files found in 'Data/Resumes'.")
    logging.info("Reading from 'Data/Resumes' is now complete.")
except FileNotFoundError as e:
    logging.error(str(e))
    logging.error("Exiting from the program.")
    logging.error("Please add resumes in the 'Data/Resumes' folder and try again.")
    sys.exit(1)
except Exception as e:
    logging.error(f"An unexpected error occurred: {str(e)}")
    sys.exit(1)

# Now after getting the file_names parse the resumes into a JSON Format.
logging.info("Started parsing the resumes.")
for file in file_names:
    processor = ResumeProcessor(file)
    success = processor.process()
    if success:
        logging.info(f"Successfully processed resume: {file}")
    else:
        logging.warning(f"Failed to process resume: {file}")
logging.info("Parsing of the resumes is now complete.")

# Process Job Descriptions
logging.info("Started to read from 'Data/JobDescription'")
try:
    remove_old_files(PROCESSED_JOB_DESCRIPTIONS_PATH)
    file_names = get_filenames_from_dir("Data/JobDescription")
    if not file_names:
        raise FileNotFoundError(
            "No job description files found in 'Data/JobDescription'."
        )
    logging.info("Reading from 'Data/JobDescription' is now complete.")
except FileNotFoundError as e:
    logging.error(str(e))
    logging.error("Exiting from the program.")
    logging.error(
        "Please add job descriptions in the 'Data/JobDescription' folder and try again."
    )
    sys.exit(1)
except Exception as e:
    logging.error(f"An unexpected error occurred: {str(e)}")
    sys.exit(1)

# Now after getting the file_names parse the job descriptions into a JSON Format.
logging.info("Started parsing the Job Descriptions.")
for file in file_names:
    processor = JobDescriptionProcessor(file)
    success = processor.process()
    if success:
        logging.info(f"Successfully processed job description: {file}")
    else:
        logging.warning(f"Failed to process job description: {file}")
logging.info("Parsing of the Job Descriptions is now complete.")
logging.info("Success! Now run `streamlit run streamlit_second.py`")
