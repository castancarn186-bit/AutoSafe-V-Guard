from abc import ABC, abstractmethod
from core.protocol import RiskReport

class BaseGuardModule(ABC):
    """
    所有安全检测模块的抽象基类
    """
    def __init__(self, module_id: str):
        self.module_id = module_id

    @abstractmethod
    async def detect(self, context: dict) -> RiskReport:
        """
        子模块必须实现的检测逻辑
        :param context: 包含 'audio' (语音流) 和 'state' (车辆状态) 的字典
        :return: RiskReport 实例
        """
        pass