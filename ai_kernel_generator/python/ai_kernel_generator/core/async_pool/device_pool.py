# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import asyncio
from typing import List
import logging

logger = logging.getLogger(__name__)

class DevicePool:
    """
    设备资源池管理类，用于分配和回收Ascend/CUDA设备

    Args:
        device_type (str): 设备类型，支持 'ascend' 或 'cuda'
        device_count (int): 可用设备总数
    """

    def __init__(self, device_list: List[str]):
        self.available_devices = asyncio.Queue()
        self.condition = asyncio.Condition()
        self.device_list = device_list
        
        # 初始化设备ID队列
        for device_id in device_list:
            self.available_devices.put_nowait(device_id)

    async def acquire_device(self) -> str:
        """
        异步获取可用设备

        Returns:
            str: 设备ID

        Raises:
            RuntimeError: 当设备类型不匹配时抛出
        """
        async with self.condition:
            while self.available_devices.empty():
                await self.condition.wait()
            device_id = await self.available_devices.get()
            logger.debug(f"Acquired device: {device_id}")
            return device_id

    async def release_device(self, device_id: str):
        """
        释放设备回资源池

        Args:
            device_id (str): 要释放的设备ID
        """
        async with self.condition:
            await self.available_devices.put(device_id)
            self.condition.notify()
            logger.debug(f"Released device: {device_id}")
