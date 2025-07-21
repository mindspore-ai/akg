import pytest
from pathlib import Path
from ai_kernel_generator import get_project_root
from ai_kernel_generator.database.database_rag import DatabaseRAG
from ..utils import get_benchmark_name

DEFAULT_BENCHMARK_PATH = Path(get_project_root()).parent.parent / "benchmark"

def gen_task_code(framework_path: str = "", impl_path: str = ""):
    framework_code = ""
    impl_code = ""
    if framework_path:
        framework_code = Path(framework_path).read_text()
    if impl_path:
        impl_code = Path(impl_path).read_text()
    return impl_code, framework_code


@pytest.mark.level0
def test_insert():
    # 初始化系统
    config_path = Path(get_project_root()) / "database" / "rag_config.yaml"
    db_system = DatabaseRAG(str(config_path))

    arch = "ascend910b4"
    backend="ascend"
    framework = "numpy"
    impl_type="triton"

    # 插入示例
    id_list = [2, 3, 4, 5]
    benchmark_name = get_benchmark_name(id_list)
    for op_name in benchmark_name:
        impl_code, framework_code = gen_task_code(
            framework_path=DEFAULT_BENCHMARK_PATH / "kernelbench" / framework / op_name / f"{op_name}_{framework}.py",
            impl_path=DEFAULT_DATABASE_PATH / "triton" / arch / f"{op_name}_{impl_type}.py"
        )
        db_system.insert(impl_code, framework_code, backend, arch, impl_type, framework)


@pytest.mark.level0
def test_sample():
    # 初始化系统
    config_path = Path(get_project_root()) / "database" / "rag_config.yaml"
    db_system = DatabaseRAG(str(config_path), top_k=3)

    arch = "ascend910b4"
    backend="ascend"
    framework = "numpy"
    impl_type="triton"

    # 查询示例
    op_name = get_benchmark_name([1])[0]
    query_code_path = DEFAULT_DATABASE_PATH / "triton" / arch / f"{op_name}_{impl_type}.py"
    query_code = Path(query_code_path).read_text()
    recall, results = db_system.sample(query_code, backend=backend, arch=arch, impl_type=impl_type)
    print("===================================================")
    print(f"找到 {len(results)} 个匹配的优化方案:")
    for i, res in enumerate(results, 1):
        print(f"\n#{i} 距离: {res['similarity_score']:.4f}")
        print(f"算子名称: {res['operator_name']}")
        print(f"文件路径: {res['file_path']}")
    print(f"\n召回率: {recall:.4f}")
    print("===================================================")


@pytest.mark.level0
def test_delete():
    # 初始化系统
    config_path = Path(get_project_root()) / "database" / "rag_config.yaml"
    db_system = DatabaseRAG(str(config_path))

    arch = "ascend910b4"
    backend="ascend"
    framework = "numpy"
    impl_type="triton"

    # 删除示例
    op_name = get_benchmark_name([3])[0]
    impl_path = DEFAULT_DATABASE_PATH / "triton" / arch / f"{op_name}_{impl_type}.py"
    impl_code = Path(impl_path).read_text()
    db_system.delete(impl_code, backend, arch, impl_type)