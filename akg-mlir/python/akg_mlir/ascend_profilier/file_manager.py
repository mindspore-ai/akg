# Copyright 2023-2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""common utils for file manager"""
import os
import csv
import shutil
import logging

MAX_FILE_SIZE = 1024 * 1024 * 1024 * 10
MAX_CSV_SIZE = 1024 * 1024 * 1024 * 5


class FileManager:
    """Files Manager."""
    @classmethod
    def file_read_all(cls, file_path: str, mode: str = "r") -> any:
        """read all files."""
        if not os.path.isfile(file_path):
            return ''
        file_size = os.path.getsize(file_path)
        if file_size <= 0:
            return ''
        if file_size > MAX_FILE_SIZE:
            logging.warning("The file size exceeds the preset value %sMB, please check the file: %s",
                            MAX_FILE_SIZE / 1024 / 1024, file_path)
            return ''
        try:
            with open(file_path, mode, encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise RuntimeError(f"Can't read file: {file_path}") from e

    @classmethod
    def read_csv_file(cls, file_path: str, class_bean: any) -> list:
        """Read CSV file and parse rows into bean objects."""
        if not os.path.isfile(file_path):
            return []
        file_size = os.path.getsize(file_path)
        if file_size <= 0:
            return []
        if file_size > MAX_CSV_SIZE:
            logging.warning("The file size exceeds the preset value %sMB, please check the file: %s",
                            MAX_CSV_SIZE / 1024 / 1024, file_path)
            return []
        result_data = []
        try:
            with open(file_path, newline="", encoding='utf-8') as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    result_data.append(class_bean(row))
        except Exception as e:
            raise RuntimeError(f"Failed to read the file: {file_path}") from e
        return result_data

    @classmethod
    def remove_file_safety(cls, path: str):
        """Safely remove a file or directory."""
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
            except Exception as e:
                logging.error("Can't remove the directory: %s", path)
                raise RuntimeError(f"Failed to remove the file: {file_path}") from e
