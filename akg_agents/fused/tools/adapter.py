"""
工具适配层: 将 demo 的 ToolRegistry 工具桥接为 akg_agents v2 兼容的工具。

设计思路:
- demo 的工具定义（JSON Schema + execute 函数）已经很好，不需要重写
- 只需要一个薄适配层，让 akg_agents v2 的 ToolExecutor 能调用它们
- 保持 demo 工具的所有能力（AST 分析、依赖追踪、验证等）

使用方式:
    # 在 akg_agents v2 agent 中注册 demo 工具
    from fused.tools.adapter import create_kernelbench_tools
    tools = create_kernelbench_tools(workspace_dir="/path/to/workspace")

适配模式:
    demo ToolRegistry                akg_agents v2
    ┌──────────────┐    adapter     ┌──────────────┐
    │ register()   │  ──────────>   │ domain tools │
    │ execute()    │                │ (函数列表)    │
    │ list_for_    │  <──────────   │              │
    │   prompt()   │    调用        │ ToolExecutor │
    └──────────────┘                └──────────────┘
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional

# 确保 demo 模块可导入
_DEMO_ROOT = Path(__file__).resolve().parent.parent
_AKG_ROOT = _DEMO_ROOT.parent
_PYTHON_ROOT = _AKG_ROOT / "python"

for p in [str(_AKG_ROOT), str(_PYTHON_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


class ToolAdapter:
    """
    单个工具的适配包装器。
    将 demo 的 {id, description, parameters, execute} 转换为
    akg_agents v2 兼容的可调用对象 + 元数据。
    """

    def __init__(self, tool_id: str, description: str,
                 parameters: Dict, execute_fn: Callable):
        self.tool_id = tool_id
        self.description = description
        self.parameters = parameters  # JSON Schema
        self._execute_fn = execute_fn

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具，返回标准格式结果"""
        return self._execute_fn(arguments)

    def to_v2_tool_info(self) -> Dict[str, Any]:
        """转换为 akg_agents v2 的工具描述格式"""
        return {
            "name": self.tool_id,
            "description": self.description,
            "parameters": self.parameters,
            "type": "domain",  # akg_agents v2 tool type
        }

    def to_prompt_description(self) -> str:
        """生成适合放入 LLM prompt 的工具描述"""
        params = self.parameters.get("properties", {})
        required = self.parameters.get("required", [])

        param_lines = []
        for name, info in params.items():
            ptype = info.get("type", "any")
            pdesc = info.get("description", "")
            req = " (必填)" if name in required else ""
            param_lines.append(f"    - {name}: {ptype}{req} - {pdesc}")

        params_str = "\n".join(param_lines) if param_lines else "    (无参数)"
        return f"- **{self.tool_id}**: {self.description}\n  参数:\n{params_str}"


class KernelBenchToolkit:
    """
    KernelBench 领域工具集。
    封装 demo 的所有工具，提供统一的注册和执行接口。

    与 akg_agents v2 集成:
    - 作为 domain tools 注册到 ToolExecutor
    - 或在 Agent 初始化时加载
    """

    def __init__(self, workspace_dir: Optional[str] = None,
                 output_dir: Optional[str] = None):
        """
        Args:
            workspace_dir: 工作区目录（存储提取的代码片段）
            output_dir: 输出目录（存储生成的任务文件）
        """
        self._tools: Dict[str, ToolAdapter] = {}
        self._workspace_dir = workspace_dir
        self._output_dir = output_dir
        self._initialized = False

    def _ensure_initialized(self):
        """延迟初始化 demo 工具（避免导入时的副作用）"""
        if self._initialized:
            return

        # 配置 demo 的目录（如果指定了自定义路径）
        if self._workspace_dir or self._output_dir:
            self._configure_demo_paths()

        # 导入 demo 的 ToolRegistry（触发工具注册）
        from demo.tools.registry import ToolRegistry
        # 触发工具注册
        import demo.tools.code_tools   # noqa: F401
        import demo.tools.file_tools   # noqa: F401
        import demo.tools.user_tools   # noqa: F401

        # 将 demo 工具包装为 ToolAdapter
        for tool_id, tool_info in ToolRegistry._tools.items():
            self._tools[tool_id] = ToolAdapter(
                tool_id=tool_id,
                description=tool_info["description"],
                parameters=tool_info["parameters"],
                execute_fn=tool_info["execute"],
            )

        self._initialized = True

    def _configure_demo_paths(self):
        """配置 demo 的工作目录"""
        import demo.config as demo_config
        if self._workspace_dir:
            demo_config.WORKSPACE_DIR = Path(self._workspace_dir)
            demo_config.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        if self._output_dir:
            demo_config.OUTPUT_DIR = Path(self._output_dir)
            demo_config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行指定工具。

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            {status: "success"|"error", output: str, error: str}
        """
        self._ensure_initialized()
        tool = self._tools.get(tool_name)
        if not tool:
            return {
                "status": "error",
                "output": "",
                "error": f"未知工具: {tool_name}。可用工具: {list(self._tools.keys())}"
            }
        try:
            return tool.execute(arguments)
        except Exception as e:
            return {
                "status": "error",
                "output": "",
                "error": f"工具 {tool_name} 执行异常: {e}"
            }

    def list_tools(self) -> List[str]:
        """列出所有可用工具名"""
        self._ensure_initialized()
        return list(self._tools.keys())

    def get_tool(self, tool_name: str) -> Optional[ToolAdapter]:
        """获取工具适配器实例"""
        self._ensure_initialized()
        return self._tools.get(tool_name)

    def get_tools_for_prompt(self) -> str:
        """生成适合放入 LLM prompt 的工具列表描述"""
        self._ensure_initialized()
        parts = []
        for tool in self._tools.values():
            parts.append(tool.to_prompt_description())
        return "\n\n".join(parts)

    def get_v2_tool_infos(self) -> List[Dict[str, Any]]:
        """获取所有工具的 v2 格式描述（用于 ToolExecutor 注册）"""
        self._ensure_initialized()
        return [t.to_v2_tool_info() for t in self._tools.values()]

    # ---- 领域专用工具的便捷方法 ----

    def assemble_task(self, **kwargs) -> Dict[str, Any]:
        """装配 KernelBench 任务"""
        return self.execute("assemble_task", kwargs)

    def trace_dependencies(self, file_path: str,
                           entry_functions: List[str]) -> Dict[str, Any]:
        """追踪函数依赖"""
        return self.execute("trace_dependencies", {
            "file_path": file_path,
            "entry_functions": entry_functions,
        })

    def validate_task(self, task_file: str, **kwargs) -> Dict[str, Any]:
        """验证任务格式"""
        return self.execute("validate_task", {"task_file": task_file, **kwargs})

    def test_with_reference(self, reference_code: str,
                            **kwargs) -> Dict[str, Any]:
        """参考对比测试"""
        return self.execute("test_with_reference", {
            "reference_code": reference_code, **kwargs
        })


def create_kernelbench_tools(
    workspace_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> KernelBenchToolkit:
    """
    创建 KernelBench 领域工具集。

    在 akg_agents v2 中的使用方式:

    ```python
    # 在 Agent 的 _load_available_tools() 中:
    toolkit = create_kernelbench_tools(workspace_dir="./workspace")

    # 获取工具描述（注入 prompt）
    tools_prompt = toolkit.get_tools_for_prompt()

    # 执行工具
    result = toolkit.execute("trace_dependencies", {
        "file_path": "workspace/decompositions.py",
        "entry_functions": ["_chunk_cat"]
    })
    ```
    """
    return KernelBenchToolkit(
        workspace_dir=workspace_dir,
        output_dir=output_dir,
    )
