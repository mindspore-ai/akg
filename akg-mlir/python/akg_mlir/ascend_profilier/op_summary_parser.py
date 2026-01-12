"""Op summary parser module."""
from .op_summary_headers import OpSummaryHeaders
from .file_manager import FileManager
from .op_summary_bean import OpSummaryBean
from .cann_file_parser import CANNFileParser, CANNDataEnum
from .path_manager import PathManager

MAX_TIME_CONSUME = 9999999999

class OpSummaryParser():
    """Parser for operation summary data."""

    def __init__(self, profiler_path: str, arch: dict):
        self._profiler_path = profiler_path
        self._cann_path = PathManager.get_cann_path(profiler_path)
        self._arch = arch

    @classmethod
    def _project_map_for_headers(cls, input_headers: list):
        """Map input headers to kernel base headers."""
        project_map_dict = {
            OpSummaryHeaders.OP_SUMMARY_SHOW_HEADERS[i]: OpSummaryHeaders.OP_SUMMARY_KERNEL_BASE_HEADERS[i]
            for i in range(len(OpSummaryHeaders.OP_SUMMARY_SHOW_HEADERS))
        }
        output_headers = []
        for header in input_headers:
            if header in project_map_dict:
                output_headers.append(project_map_dict.get(header))
            else:
                output_headers.append(header)
        return output_headers

    def generate_op_summary_data(self):
        """Generate operation summary data from profiling files."""
        op_summary_file_set = CANNFileParser(self._profiler_path).get_file_list_by_type(CANNDataEnum.OP_SUMMARY)
        summary_data = []
        for file_path in op_summary_file_set:
            all_data = FileManager.read_csv_file(file_path, OpSummaryBean)
            if all_data:
                OpSummaryBean.headers = OpSummaryHeaders.OP_SUMMARY_SHOW_HEADERS
                summary_data.extend([data.row for data in all_data])
            else:
                raise RuntimeError("parse op summary csv failed.")
        task_duration_list = []
        for line in summary_data:
            if line[2] == 'AI_VECTOR_CORE':
                task_duration_list.append(float(line[4]))
        task_duration_list.sort()
        task_duration_list = task_duration_list[1:-1]
        task_duration = sum(task_duration_list) / len(task_duration_list)
        return task_duration

    def check_task_duration(self, csv_file_row):
        """Check if task duration is valid."""
        cur_task_duration = float(csv_file_row.get(OpSummaryHeaders.TASK_DURATION, MAX_TIME_CONSUME))
        is_res = cur_task_duration > 0
        if self._arch is not None and "910B" in self._arch:
            cur_task_type = str(csv_file_row.get(OpSummaryHeaders.TASK_TYPE, ""))
            is_res = is_res and cur_task_type in ('AI_CORE', 'AI_VECTOR_CORE')
        return is_res

    def create_dict(self, summary_data, headers):
        """Create dictionary from summary data and headers."""
        summary_dict = {}
        for i, data in enumerate(summary_data):
            summary_dict[headers[i]] = data
        return summary_dict
