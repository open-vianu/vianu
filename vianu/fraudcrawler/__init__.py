import logging

# Create a named logger
logger = logging.getLogger("fraudcrawler_logger")

# Set the log level
logger.setLevel(logging.INFO)

# Define a custom formatter
formatter = logging.Formatter(
    fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)
