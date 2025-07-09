# DevicePool 模块设计文档

## 概述
DevicePool模块负责Ascend, CUDA, CPU等设备的异步分配与回收管理，采用生产者-消费者模式实现设备资源的并发控制。通过asyncio.Condition实现线程安全的设备等待通知机制。

## 初始化参数
| 参数名称 | 类型/必选 | 参数说明 |
|---------|---------|---------|
| device_list | List[int] (必选) | 设备ID列表（如[0, 1]）|

## 类方法说明
### `acquire_device()`

**功能**
- 异步获取一个可用的设备。

**参数说明**
| 参数名 | 类型 | 说明 |
|-------|-----|-----|
| 返回值 | int | 设备ID（格式：0, 1） |


### `release_device()`

**功能**
- 异步释放指定设备。

**参数说明**
| 参数名 | 类型 | 说明 |
|-------|-----|-----|
| device_id | int | 需释放的设备ID |



## 使用示例
```python
from ai_kernel_generator.core.async_pool import DevicePool
import asyncio

# 初始化含3个Ascend设备的资源池
device_pool = DevicePool([0, 1, 2])

async def kernel_task():
    device_id = await device_pool.acquire_device()
    try:
        # 模拟设备计算任务
        print(f"Start processing on {device_id}")
        await asyncio.sleep(1)
        return f"Task completed on {device_id}"
    finally:
        await device_pool.release_device(device_id)

async def main():
    # 并发执行5个任务（资源池自动调度）
    tasks = [kernel_task() for _ in range(5)]
    results = await asyncio.gather(*tasks)
    print("Execution results:", results)

if __name__ == "__main__":
    asyncio.run(main())
```
