from abc import ABC, abstractmethod
import sys
from pathlib import Path

# 确保能导入项目根目录的 core 模块
current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# 不再需要导入 DetectionResult，因为使用了字符串注解
# from core.protocol import DetectionResult

class BaseGuardModule(ABC):
    """
    所有安全检测模块的抽象基类
    """
    def __init__(self, module_id: str):
        self.module_id = module_id

    @abstractmethod
    async def detect(self, context: dict) -> 'DetectionResult':
        """
        子模块必须实现的检测逻辑
        :param context: 包含 'audio' (语音流) 和 'state' (车辆状态) 的字典
        :return: RiskReport 实例
        """
        pass