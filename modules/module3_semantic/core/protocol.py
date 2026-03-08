"""
V-Guard 核心数据契约模块 (基于 Pydantic 重构)
作用：严苛的运行时数据校验、多模块解耦、快速序列化
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from enum import Enum
import time
import uuid

# --- 枚举类型保持清晰 ---
class Language(str, Enum):
    ZH = "zh"
    EN = "en"

class WeatherCondition(str, Enum):
    SUNNY = "sunny"
    RAINY = "rainy"
    FOGGY = "foggy"
    SNOWY = "snowy"

class RiskLevel(str, Enum):
    SAFE = "SAFE"           # 0.0 - 0.3
    WARNING = "WARNING"     # 0.3 - 0.7
    DANGER = "DANGER"       # 0.7 - 1.0

class IntentCategory(str, Enum):
    HVAC = "hvac"                 # 空调温度控制
    BODY_CONTROL = "body_control" # 车窗、车门、后备箱、雨刮
    ENTERTAINMENT = "entertainment" # 视频、音乐
    NAVIGATION = "navigation"     # 导航
    SAFETY = "safety"             # 紧急制动

# --- 数据契约类 (引入强大的约束) ---

class VehicleContext(BaseModel):
    """车况上下文：加入了边界约束，防止脏数据"""
    speed: float = Field(..., ge=0, le=300, description="当前车速 km/h (0-300)")
    speed_limit: float = Field(..., ge=10, le=120, description="道路限速")
    gear: str = Field(..., pattern="^(P|R|N|D)$", description="档位限制为PRND")
    weather: WeatherCondition
    traffic_density: str = Field(..., pattern="^(low|medium|high)$")
    has_pedestrians: bool = Field(default=False, description="周围是否有行人")
    window_open: bool = Field(default=False, description="当前车窗是否开启")
    
    @field_validator('speed')
    def check_speed_and_gear(cls, v, info):
        # 联动校验：如果是P档，速度必须是0
        if 'gear' in info.data and info.data['gear'] == 'P' and v > 0:
            raise ValueError("P档下车速必须为0")
        return v

class SemanticInput(BaseModel):
    """语义层输入：ASR的输出和当前车况的集合"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., min_length=1, max_length=500, description="ASR文本")
    language: Language
    context: VehicleContext
    timestamp: float = Field(default_factory=time.time)

class RiskReport(BaseModel):
    """语义层输出：最终的评估结果"""
    risk_score: float = Field(..., ge=0.0, le=1.0, description="风险打分")
    level: RiskLevel
    reason: str = Field(..., description="给用户的可解释拒绝原因")
    intent_category: IntentCategory
    matched_vector_id: Optional[str] = Field(None, description="命中的HNSW向量节点ID，若是深度学习推理则为None")

class VehicleContext(BaseModel):
    speed: float = Field(default=0.0, ge=0, le=300)
    # 👈 修改：增加默认值，防止因为缺失该字段报错
    speed_limit: float = Field(default=120.0, ge=10, le=120) 
    gear: str = Field(default="P", pattern="^(P|R|N|D)$")
    weather: WeatherCondition = WeatherCondition.SUNNY
    traffic_density: str = Field(default="low", pattern="^(low|medium|high)$")
    has_pedestrians: bool = Field(default=False)