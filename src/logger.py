import logging
from datetime import datetime
import os

# 1. Define log file name with current time and date
log_file_name = f"{datetime.now().strftime('%d%m%y__%H%M%S')}.log"

# 2. Define log directory path
log_dir = os.path.join(os.getcwd(), "logs")

# 3. Create 'logs' directory in current working directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# 4. Full log file path
log_file_path = os.path.join(log_dir, log_file_name)

# 5. Configure logging
logging.basicConfig(
    filename=log_file_path,
    format='[%(asctime)s] Line: %(lineno)d | %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logging.getLogger("numexpr.utils").setLevel(logging.WARNING)