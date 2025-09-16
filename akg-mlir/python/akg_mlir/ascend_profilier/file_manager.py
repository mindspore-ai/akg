import csv
import json
import os.path
import shutil
from warnings import warn

MAX_FILE_SIZE = 1024 * 1024 * 1024 * 10
MAX_CSV_SIZE = 1024 * 1024 * 1024 * 5

class FileManager:
    @classmethod
    def file_read_all(cls, file_path: str, mode: str = "r") -> any:
        if not os.path.isfile(file_path):
            return ''
        file_size = os.path.getsize(file_path)
        if file_size <= 0:
            return ''
        if file_size > MAX_FILE_SIZE:
            warn(f"The file size exceeds the preset value {MAX_FILE_SIZE / 1024 / 1024}MB, "
                 f"please check the file: {file_path}")
            return ''
        try:
            with open(file_path, mode) as file:
                return file.read()
        except Exception:
            raise RuntimeError(f"Can't read file: {file_path}")

    @classmethod
    def read_csv_file(cls, file_path: str, class_bean: any) -> list:
        if not os.path.isfile(file_path):
            return []
        file_size = os.path.getsize(file_path)
        if file_size <= 0:
            return []
        if file_size > MAX_CSV_SIZE:
            warn(f"The file size exceeds the preset value {MAX_CSV_SIZE / 1024 / 1024}MB, "
                 f"please check the file: {file_path}")
            return []
        result_data = []
        try:
            with open(file_path, newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    result_data.append(class_bean(row))
        except Exception:
            raise RuntimeError(f"Failed to read the file: {file_path}")
        return result_data

    @classmethod
    def remove_file_safety(cls, path: str):
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
            except Exception:
                print(f"[WARNING] [{os.getpid()}] profiler.py: Can't remove the directory: {path}")

