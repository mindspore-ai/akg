import os
import re

class PathManager:

    @classmethod
    def get_cann_path(cls, profiler_path: str) -> str:
        sub_dirs = os.listdir(os.path.realpath(profiler_path))
        sub_dirs.sort(reverse=True)
        for sub_dir in sub_dirs:
            sub_path = os.path.join(profiler_path, sub_dir)
            if os.path.isdir(sub_path) and (re.match(r"^PROF_\d+_\d+_\d+[a-zA-Z]+", sub_dir) or re.match(r"^PROF_\d+_\d+[a-zA-Z]+", sub_dir)):
                return sub_path
        return ""

    @classmethod
    def get_device_path(cls, cann_path: str) -> str:
        sub_dirs = os.listdir(os.path.realpath(cann_path))
        for sub_dir in sub_dirs:
            sub_path = os.path.join(cann_path, sub_dir)
            if os.path.isdir(sub_path) and re.match(r"^device_\d", sub_dir):
                return sub_path
        return ""

    @classmethod
    def get_start_info_path(cls, cann_path: str) -> str:
        start_info_path = os.path.join(cann_path, "host", "start_info")
        if os.path.exists(start_info_path):
            return start_info_path
        device_path = cls.get_device_path(cann_path)
        if not device_path:
            return ""
        device_path_split = os.path.basename(device_path).split("_")
        if len(device_path_split) != 2:
            return ""
        start_info_file = f"start_info.{device_path_split[1]}"
        start_info_path = os.path.join(device_path, start_info_file)
        if os.path.exists(start_info_path):
            return start_info_path
        return ""

    @classmethod
    def get_device_all_file_list_by_type(cls, profiler_path: str, summary_or_timeline: str) -> list:
        file_list = []
        _path = os.path.join(cls.get_device_path(profiler_path), summary_or_timeline)
        if not os.path.isdir(_path):
            return file_list
        sub_files = os.listdir(os.path.realpath(_path))
        if not sub_files:
            return file_list
        for sub_file in sub_files:
            file_list.append(os.path.join(_path, sub_file))
        return file_list

    @classmethod
    def get_analyze_all_file(cls, profiler_path: str, analyze: str) -> list:
        file_list = []
        _path = os.path.join(profiler_path, analyze)
        if not os.path.isdir(_path):
            return file_list
        sub_files = os.listdir(os.path.realpath(_path))
        if not sub_files:
            return file_list
        for sub_file in sub_files:
            file_list.append(os.path.join(_path, sub_file))
        return file_list
