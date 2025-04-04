# import logging
# import os
# from logging import getLogger, StreamHandler, FileHandler, Formatter
# from colorama import init, Fore, Style

# # Initialize colorama for Windows compatibility
# init(autoreset=True)

# class ColoredFormatter(Formatter):
#     """Custom formatter to add colors to log levels for console output."""

#     COLORS = {
#         logging.DEBUG: Fore.BLUE + Style.BRIGHT,     # Blue for DEBUG
#         logging.INFO: Fore.GREEN + Style.BRIGHT,     # Green for INFO
#         logging.WARNING: Fore.YELLOW + Style.BRIGHT, # Yellow for WARNING
#         logging.ERROR: Fore.RED + Style.BRIGHT,      # Red for ERROR
#         logging.CRITICAL: Fore.MAGENTA + Style.BRIGHT # Magenta for CRITICAL
#     }

#     def format(self, record):
#         color = self.COLORS.get(record.levelno, "")
#         reset = Style.RESET_ALL
#         record.msg = f"{color}{record.msg}{reset}"
#         return super().format(record)

# def get_logger(filename):
#     """Setup logger with color formatting for console and plain text for files."""
#     logger = getLogger(__name__)
#     logger.setLevel(logging.DEBUG)  # Capture all log levels

#     # Ensure the log file exists
#     if not os.path.exists(filename):
#         with open(filename, "w"):  # Create an empty file
#             pass

#     # Prevent multiple handlers from being added on repeated calls
#     if not logger.hasHandlers():
#         # Console handler (with color formatting)
#         console_handler = StreamHandler()
#         console_handler.setFormatter(ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s"))

#         # File handler (without color formatting)
#         file_handler = FileHandler(filename, mode='a')
#         file_handler.setFormatter(Formatter("%(asctime)s - %(levelname)s - %(message)s"))

#         # Add handlers
#         logger.addHandler(console_handler)
#         logger.addHandler(file_handler)

#     return logger


import logging
import os
from logging import getLogger, StreamHandler, FileHandler, Formatter
from colorama import init, Fore, Style

# Initialize colorama for Windows compatibility
init(autoreset=True)

class ColoredFormatter(Formatter):
    """Custom formatter to add colors to log levels for console output."""

    COLORS = {
        logging.DEBUG: Fore.BLUE + Style.BRIGHT,     # Blue for DEBUG
        logging.INFO: Fore.GREEN + Style.BRIGHT,     # Green for INFO
        logging.WARNING: Fore.YELLOW + Style.BRIGHT, # Yellow for WARNING
        logging.ERROR: Fore.RED + Style.BRIGHT,      # Red for ERROR
        logging.CRITICAL: Fore.MAGENTA + Style.BRIGHT # Magenta for CRITICAL
    }

    def format(self, record):
        """Apply color only for console output."""
        color = self.COLORS.get(record.levelno, "")
        reset = Style.RESET_ALL
        original_msg = record.msg  # Store the original message
        record.msg = f"{color}{record.msg}{reset}"  # Apply color
        formatted_message = super().format(record)
        record.msg = original_msg  # Restore original message for other handlers
        return formatted_message

def get_logger(filename):
    """Setup logger with color formatting for console and plain text for files."""
    logger = getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Capture all log levels

    # Ensure the log file exists
    if not os.path.exists(filename):
        with open(filename, "w"):  # Create an empty file
            pass

    # Prevent duplicate handlers if logger is called multiple times
    if not logger.hasHandlers():
        # Console handler (with color formatting)
        console_handler = StreamHandler()
        console_handler.setFormatter(ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s"))

        # File handler (without color formatting)
        file_handler = FileHandler(filename, mode='a')
        file_handler.setFormatter(Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger