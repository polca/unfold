"""
This module contains constants for the filesystem paths used by Unfold.
"""

from pathlib import Path
import platformdirs


DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Directories for user-created data
USER_DATA_BASE_DIR = platformdirs.user_data_path(
        appname="unfold", appauthor="polca"
    )
USER_DATA_BASE_DIR.mkdir(parents=True, exist_ok=True)

DIR_CACHED_DB = USER_DATA_BASE_DIR / "cache"
DIR_CACHED_DB.mkdir(parents=True, exist_ok=True)

USER_LOGS_DIR = platformdirs.user_log_path(appname="unfold", appauthor="polca")
USER_LOGS_DIR.mkdir(parents=True, exist_ok=True)
