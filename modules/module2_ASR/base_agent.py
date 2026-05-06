# base_agent.py
"""
智能体基类
所有 Agent 都继承此类，通过消息总线通信
"""

import asyncio
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """智能体基类"""

    def __init__(self, name: str, bus):
        self.name = name
        self.bus = bus
        self._subscriptions = []

        # 自动订阅（子类实现 subscribe_all）
        self.subscribe_all()

    @abstractmethod
    async def subscribe_all(self):
        """子类重写此方法，订阅需要的 topic"""
        pass

    async def publish(self, topic: str, payload: dict):
        """发布消息到总线"""
        if hasattr(self.bus, 'publish'):
            await self.bus.publish(topic, sender=self.name, payload=payload)
        else:
            # 如果 bus 不是异步的，提供同步包装
            print(f"[{self.name}] -> {topic}: {payload}")

    async def subscribe(self, topic: str, handler):
        """订阅消息（由子类调用）"""
        if hasattr(self.bus, 'subscribe'):
            await self.bus.subscribe(topic, handler)
        self._subscriptions.append((topic, handler))

    @abstractmethod
    async def start(self):
        """启动 Agent"""
        pass

    async def stop(self):
        """停止 Agent（可选重写）"""
        pass