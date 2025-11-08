"""
File Validator - Validates existence of required files
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FileValidator:
    """Validates required files for the VLSI flow"""

    @staticmethod
    def validate_benchmark_files(file_paths):
        """
        Verify all required input files exist

        Args:
            file_paths: Dictionary of file paths to validate

        Returns:
            bool: True if all files exist, False otherwise
        """
        logger.info("Checking input files...")

        missing = []
        for file_type, file_path in file_paths.items():
            if not file_path.exists():
                missing.append(f"{file_type}: {file_path}")

        if missing:
            logger.error("Missing required files:")
            for f in missing:
                logger.error(f"  - {f}")
            return False

        logger.info("All required files found")
        return True

    @staticmethod
    def find_existing_file(possible_paths):
        """
        Find the first existing file from a list of possible paths

        Args:
            possible_paths: List of Path objects to check

        Returns:
            Path object if found, None otherwise
        """
        for path in possible_paths:
            if path.exists():
                return path
        return None

    @staticmethod
    def validate_file_exists(file_path, description="File"):
        """
        Validate a single file exists

        Args:
            file_path: Path to the file
            description: Description of the file for error messages

        Returns:
            bool: True if file exists, False otherwise
        """
        if not file_path.exists():
            logger.error(f"{description} not found: {file_path}")
            return False

        logger.info(f"{description} found: {file_path}")
        return True

