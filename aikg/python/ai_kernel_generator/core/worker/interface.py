from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict, Union

class WorkerInterface(ABC):
    """
    Abstract base class for AIKG Workers (Local and Remote).
    """

    @abstractmethod
    async def verify(self, package_data: Union[bytes, str], task_id: str, op_name: str, timeout: int = 300) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Execute verification task.
        
        注意：device 的管理（acquire/release）由调用方负责
        worker 只负责执行已经生成好的脚本（脚本中已包含正确的 device_id）

        Args:
            package_data: Verification project内容。可为TAR字节流（远程/本地通用）
                或本地目录路径（仅LocalWorker直接复用现有目录时使用）。
            task_id: Unique task identifier.
            op_name: Operator name.
            timeout: Execution timeout in seconds.

        Returns:
            Tuple[bool, str, Dict[str, Any]]: (success, log_output, artifacts)
            - success: 验证是否成功
            - log_output: 执行日志
            - artifacts: 执行过程中生成的文件内容，格式为 {relative_path: json_content}
              例如: {"autotune_info_case_0.json": {...}, "subdir/result.json": {...}}
        """
        pass

    @abstractmethod
    async def profile(self, package_data: bytes, task_id: str, op_name: str, profile_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute profiling task.

        Args:
            package_data: The compressed verification project (TAR bytes).
            task_id: Unique task identifier.
            op_name: Operator name.
            profile_settings: Settings for profiling (e.g., warmup_times, run_times).

        Returns:
            Dict[str, Any]: Profiling results, including:
                - gen_time: 生成代码执行时间
                - base_time: 基准代码执行时间
                - speedup: 加速比
                - artifacts: 执行过程中生成的文件内容，格式为 {relative_path: json_content}
        """
        pass
