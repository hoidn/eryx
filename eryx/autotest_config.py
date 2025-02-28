import os
import logging
from eryx.autotest.configuration import Configuration

# Create configuration with debug mode enabled
config = Configuration(debug=True, log_file_prefix="ground_truth_data")

# Ensure log directory exists
os.makedirs(config.getLogFilePrefix(), exist_ok=True)

# Check if DEBUG_MODE environment variable is set
if os.environ.get("DEBUG_MODE") != "1":
    logging.warning("DEBUG_MODE environment variable is not set to '1'. "
                   "This may cause the debug decorators to be inactive. "
                   "Set DEBUG_MODE=1 before running the code.")
