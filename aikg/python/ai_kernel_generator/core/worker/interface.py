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

    @abstractmethod
    async def generate_reference(self, package_data: bytes, task_id: str, op_name: str, timeout: int = 120) -> Tuple[bool, str, bytes]:
        """
        Execute task_desc and generate reference data.
        
        用于 CUDA-to-Ascend 转换场景：在 GPU Worker 上执行 Triton-CUDA 代码，
        保存输出作为参考数据（.pt 文件），供 NPU Worker 验证转换后的代码正确性。

        Args:
            package_data: The compressed project (TAR bytes) containing reference.py and verify script.
            task_id: Unique task identifier.
            op_name: Operator name.
            timeout: Execution timeout in seconds.

        Returns:
            Tuple[bool, str, bytes]: (success, log_output, reference_data_bytes)
            - success: 是否成功生成参考数据
            - log_output: 执行日志
            - reference_data_bytes: .pt 文件的二进制内容（成功时），失败时为空 b''
        """
        pass

    @abstractmethod
    async def profile_single_task(self, package_data: bytes, task_id: str, op_name: str, 
                                   profile_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute single task profiling (only measure task_desc performance, no base comparison).
        
        单独测量某段代码的执行性能，不进行 base vs generation 对比。
        适用于需要单独测量某个 Model 执行时间的场景。

        Args:
            package_data: The compressed project (TAR bytes) containing profile script.
            task_id: Unique task identifier.
            op_name: Operator name.
            profile_settings: Settings for profiling (e.g., warmup_times, run_times).

        Returns:
            Dict[str, Any]: Profiling results, including:
                - time_us: 执行时间（微秒）
                - success: 是否成功
                - log: 执行日志
        """
        pass
