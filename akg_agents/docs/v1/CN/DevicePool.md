# DevicePool 模块设计文档

## 概述
DevicePool模块负责Ascend, CUDA, CPU等设备的异步分配与回收管理，采用生产者-消费者模式实现设备资源的并发控制。

## 初始化参数
| 参数名称 | 类型/必选 | 参数说明 |
|---------|---------|---------|
| device_list | List[int] (可选) | 设备ID列表（如[0, 1]），默认为[0] |

## 环境变量配置
支持通过环境变量 `AIKG_DEVICES_LIST` 设置设备列表，格式为逗号分隔的设备ID，如：`0,1,2`

## 类方法说明
### `acquire_device()`
异步获取一个可用的设备。

**返回值**
- int: 设备ID

### `release_device(device_id)`
异步释放指定设备。

**参数**
- device_id (int): 需释放的设备ID
