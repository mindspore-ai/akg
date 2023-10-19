import os
import re
import subprocess
from enum import Enum

from .file_manager import FileManager
from .path_manager import PathManager


class CANNDataEnum(Enum):
    OP_SUMMARY = 0
    NPU_MEMORY = 1
    MSPROF_TIMELINE = 2
    STEP_TRACE = 3
    GE_MEMORY_RECORD = 4
    GE_OPERATOR_MEMORY = 5
    L2_CACHE = 6
    AI_CPU = 7
    COMMUNICATION = 8
    MATRIX = 9


class CANNFileParser:
    COMMAND_SUCCESS = 0
    ACL_TO_NPU = "acl_to_npu"
    START_FLOW = "s"
    END_FLOW = "f"
    SUMMARY = "summary"
    TIMELINE = "timeline"
    ANALYZE = "analyze"
    CANN_DATA_MATCH = {
        CANNDataEnum.OP_SUMMARY: [r"^op_summary_\d+_\d+\.csv", r"^op_summary_\d+_\d+_\d+\.csv",
                                  r"^op_summary_\d+_\d+_\d+_\d+\.csv"],
    }

    def __init__(self, profiler_path: str):
        self._cann_path = PathManager.get_cann_path(profiler_path)
        self._file_dict = {}
        self._file_dispatch()

    def export_cann_profiling(self):
        if not os.path.isdir(self._cann_path):
            return
        self._del_summary_and_timeline_data()
        completed_process = subprocess.run(["msprof", "--export=on", f"--output={self._cann_path}"],
                                           capture_output=True)
        if completed_process.returncode != self.COMMAND_SUCCESS:
            raise RuntimeError(
                f"Export CANN Profiling data failed, please verify that the ascend-toolkit is installed and set-env.sh "
                f"is sourced. or you can execute the command to confirm the CANN Profiling export result: "
                f"msprof --export=on --output={self._cann_path}")


    def get_file_list_by_type(self, file_type: CANNDataEnum) -> set:
        return self._file_dict.get(file_type, set())

    def _file_dispatch(self):
        all_file_list = PathManager.get_device_all_file_list_by_type(self._cann_path, self.SUMMARY)
        all_file_list += PathManager.get_device_all_file_list_by_type(self._cann_path, self.TIMELINE)
        all_file_list += PathManager.get_analyze_all_file(self._cann_path, self.ANALYZE)
        for file_path in all_file_list:
            if not os.path.isfile(file_path):
                continue
            for data_type, re_match_exp_list in self.CANN_DATA_MATCH.items():
                for re_match_exp in re_match_exp_list:
                    if re.match(re_match_exp, os.path.basename(file_path)):
                        self._file_dict.setdefault(data_type, set()).add(file_path)

    def _del_summary_and_timeline_data(self):
        device_path = PathManager.get_device_path(self._cann_path)
        if not device_path:
            return
        summary_path = os.path.join(device_path, "summary")
        timeline_path = os.path.join(device_path, "timeline")
        FileManager.remove_file_safety(summary_path)
        FileManager.remove_file_safety(timeline_path)
