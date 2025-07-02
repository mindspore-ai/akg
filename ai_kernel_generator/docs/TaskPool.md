# TaskPool 模块设计文档

## 概述
TaskPool模块是AI Kernel Generator的异步任务管理核心组件，提供可扩展的并发任务执行能力。通过信号量机制实现任务并发控制，支持自动任务生命周期管理和异常处理。

## 初始化参数
| 参数名称 | 类型/必选 | 参数说明 |
|---------|---------|---------|
| max_concurrency | int (可选) | 最大并发任务数，默认：4 |

## 类方法说明
### create_task
**功能**：创建并启动异步任务

**参数**：
- `coro_func`: 需要执行的协程函数
- `*args`: 位置参数
- `**kwargs`: 关键字参数

**返回**：
- `asyncio.Task`: 已创建的任务对象

### wait_all
**功能**：等待所有任务完成并返回结果列表

**返回**：
- `List[Any]`: 任务执行结果列表

## 使用示例
```python
from ai_kernel_generator.core.async_pool import TaskPool
import asyncio

# 创建最大并发数为3的任务池
task_pool = TaskPool(max_concurrency=3)

async def mock_task(task_id):
    await asyncio.sleep(1)
    return f"Task {task_id} completed"

async def main():
    # 批量创建10个任务
    tasks = [
        task_pool.create_task(mock_task, i)
        for i in range(10)
    ]
    
    # 等待所有任务完成
    results = await task_pool.wait_all()
    print("All tasks results:", results)

if __name__ == "__main__":
    asyncio.run(main())
```