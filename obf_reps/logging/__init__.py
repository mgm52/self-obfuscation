from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import csv
import os
from datetime import datetime

from obf_reps.types import LoggingData

class Logger(ABC):

    @abstractmethod
    def __init__(
        self,
        log_file: Optional[str] = None,
        username: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Args:
            log_file: Where to log to.
            username: User doing the logging (required if logging is personalized,
                for example when using WandB).
            metadata: Additional key value metadata that describes the run.
        """
        ...

    @abstractmethod
    def log(self, data: Dict[str, LoggingData]) -> None: ...

    @abstractmethod
    def log_to_table(self, data: List[LoggingData], table_name: str) -> None: ...

    @abstractmethod
    def create_table(self, table_name: str, columns: List[str]) -> None: ...

    @abstractmethod
    def log_tables(self) -> None: ...

    @abstractmethod
    def log_table_name(self, table_name: str) -> None: ...

    @abstractmethod
    def print(self, msg: str) -> None: ...


class CSVTXTLogger(Logger):
    """
    Logger that:
      - Appends log_to_table() data to table_name_<timestamp>.csv
      - Appends log() data to general_logs_<timestamp>.csv
      - Appends everything (print/log/log_to_table) to full_log_<timestamp>.txt
      - Raises FileExistsError if a file with that timestamp-based name already exists.
      - Filenames are created on first usage (print/log/log_to_table).
    """

    def __init__(self,
                 log_file: Optional[str] = None,
                 username: Optional[str] = None,
                 metadata: Optional[Dict] = None,
                 print_logs_to_console: bool = True) -> None:
        super().__init__(log_file, username, metadata)
        self.timestamp = None
        self.files_created = False
        self.general_logs_file = None
        self.full_log_file = None
        self.print_logs_to_console = print_logs_to_console
        # Track which table CSV files we've created so we only do exclusive creation once
        self.created_table_files = set()

    def _create_main_files_if_needed(self):
        """
        On first usage, set filenames using current timestamp
        and create them in 'x' (exclusive) mode. Raise an error
        if they already exist.
        """
        if not self.files_created:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.general_logs_file = f"general_logs_{self.timestamp}.csv"
            self.full_log_file = f"full_log_{self.timestamp}.txt"

            # Create the CSV in exclusive mode to ensure no collisions
            with open(self.general_logs_file, "x", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "key", "value"])  # example header

            # Create the TXT in exclusive mode to ensure no collisions
            with open(self.full_log_file, "x", encoding="utf-8") as f:
                # Just write an initial line if you like:
                f.write(f"Full log started at {datetime.now()}\n")

            self.files_created = True

    def _log_to_full_txt(self, prefix: str, content: str) -> None:
        """Helper that appends a line to the full TXT log."""
        with open(self.full_log_file, "a", encoding="utf-8") as f:
            f.write(f"[{prefix}] {datetime.now()} | {content}\n")

    def print(self, msg: str) -> None:
        self._create_main_files_if_needed()
        self._log_to_full_txt("PRINT", msg)
        print(msg)  # also prints to console

    def optional_print(self, msg: str) -> None:
        self._create_main_files_if_needed()
        self._log_to_full_txt("PRINT", msg)
        if self.print_logs_to_console:
            print(msg)  # also prints to console

    def log(self, data: Dict[str, "LoggingData"]) -> None:
        self._create_main_files_if_needed()
        # Append to general logs CSV
        with open(self.general_logs_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for key, value in data.items():
                writer.writerow([datetime.now(), key, value])
        # Also append to TXT
        self._log_to_full_txt("LOG", str(data))
        if self.print_logs_to_console:
            print(f"LOG: {data}")  # also prints to console

    def log_to_table(self, data: List["LoggingData"], table_name: str) -> None:
        """
        Appends data to <table_name>_<timestamp>.csv.
        If the file does not exist yet, create it in exclusive mode ('x') to ensure no collision.
        """
        self._create_main_files_if_needed()
        csv_file = f"{table_name}_{self.timestamp}.csv"

        # If we haven't created this table file yet, do so in exclusive mode:
        if csv_file not in self.created_table_files:
            # Raise an exception if file already exists
            file_exists = os.path.exists(csv_file)
            if file_exists:
                # If this table file name is already on disk, error out
                raise FileExistsError(f"File {csv_file} already exists!")
            with open(csv_file, "x", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Write header if the first element is a dict
                if data and isinstance(data[0], dict):
                    writer.writerow(["timestamp"] + list(data[0].keys()))
            self.created_table_files.add(csv_file)

        # Now append actual data in normal append mode
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for row in data:
                if isinstance(row, dict):
                    writer.writerow([datetime.now()] + list(row.values()))
                else:
                    # Fallback if row is just a single value
                    writer.writerow([datetime.now(), row])

        # Also append to TXT
        self._log_to_full_txt("LOG_TO_TABLE", f"{table_name} | {data}")
        if self.print_logs_to_console:
            print(f"LOG_TO_TABLE: {table_name} | {data}")

    def create_table(self, table_name: str, columns: List[str]) -> None:
        """
        Creates an empty CSV with the given columns as header, in exclusive mode.
        This ensures we don't overwrite an existing file.
        """
        self._create_main_files_if_needed()
        csv_file = f"{table_name}_{self.timestamp}.csv"
        if os.path.exists(csv_file):
            raise FileExistsError(f"File {csv_file} already exists!")
        with open(csv_file, "x", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp"] + columns)
        self.created_table_files.add(csv_file)

    def log_tables(self) -> None:
        pass

    def log_table_name(self, table_name: str) -> None:
        pass
