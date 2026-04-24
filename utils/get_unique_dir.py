import os
from datetime import datetime


def get_unique_dir(base_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_dir = os.path.join(base_dir, timestamp)
    return unique_dir
