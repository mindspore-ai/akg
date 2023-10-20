from .op_summary_headers import OpSummaryHeaders
from .file_manager import FileManager
from .op_summary_bean import OpSummaryBean
from .cann_file_parser import CANNFileParser, CANNDataEnum


class OpSummaryParser():

    def __init__(self, profiler_path: str):
        self._profiler_path = profiler_path

    @classmethod
    def _project_map_for_headers(cls, input_headers: list):
        project_map_dict = {OpSummaryHeaders.OP_SUMMARY_SHOW_HEADERS[i]: OpSummaryHeaders.OP_SUMMARY_KERNEL_BASE_HEADERS[i] for i in
                            range(len(OpSummaryHeaders.OP_SUMMARY_SHOW_HEADERS))}
        output_headers = []
        for header in input_headers:
            if header in project_map_dict:
                output_headers.append(project_map_dict.get(header))
            else:
                output_headers.append(header)
        return output_headers

    def generate_op_summary_data(self) -> dict:
        op_summary_file_set = CANNFileParser(self._profiler_path).get_file_list_by_type(CANNDataEnum.OP_SUMMARY)
        summary_data = []
        for file_path in op_summary_file_set:
            all_data = FileManager.read_csv_file(file_path, OpSummaryBean)
            if all_data:
                OpSummaryBean.headers = OpSummaryHeaders.OP_SUMMARY_SHOW_HEADERS
                output_headers = OpSummaryBean.headers
                summary_data.extend([data.row for data in all_data])
            else:
                raise RuntimeError("parse op summary csv failed.")

        return self.create_dict(summary_data[0],output_headers)

    def create_dict(self,summary_data,headers):
        summary_dict={}
        for i in range(len(summary_data)):
            summary_dict[headers[i]] = summary_data[i]
        return summary_dict
