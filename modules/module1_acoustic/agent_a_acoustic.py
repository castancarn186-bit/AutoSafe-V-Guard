"""
agent_a_acoustic.py
AcousticAgent：持续监听环境声音，主动发布预警，并与指令驱动检测协同。
"""
import asyncio
import numpy as np
from typing import Dict, Any, Optional, List
from modules.module1_Acoustic.audio_stream import AudioStream
from modules.module1_Acoustic.detector import AcousticDetector

# 假设存在 BaseAgent 基类（若没有，可自行定义简单版）
class BaseAgent:
    def __init__(self, name: str, bus):
        self.name = name
        self.bus = bus
        self.logger = None  # 可替换为实际日志

    async def publish(self, topic: str, payload: Dict[str, Any]):
        """发布消息到总线"""
        # 实际实现需根据您的消息总线接口编写
        print(f"[{self.name}] Publish to {topic}: {payload}")
        # 若使用异步队列，可在此处 put

class AcousticAgent(BaseAgent):
    def __init__(self, bus, config: Optional[Dict] = None):
        super().__init__("AcousticAgent", bus)
        self.detector = AcousticDetector(config=config)
        self.detector.setup()
        self.stream = AudioStream()
        self.ambient_history: List[float] = []  # 保存最近 N 次环境风险分数
        self.history_maxlen = 5  # 保留最近 5 次检测
        self.monitor_interval = 2.0  # 每 2 秒检测一次

    async def start(self):
        """启动音频流和后台监听任务"""
        self.stream.start()
        asyncio.create_task(self._ambient_monitor())
        self.logger.info("AcousticAgent started")

    async def _ambient_monitor(self):
        """后台持续监听环境，每 2 秒检测一次，主动发布预警"""
        while True:
            await asyncio.sleep(self.monitor_interval)
            chunk = self.stream.get_chunk(chunk_samples=32000)  # 2秒音频
            if chunk is None:
                continue

            result = self.detector.detect(chunk)  # 返回字典，包含 risk_score 等
            risk = result.get("risk_score", 0.0)

            # 维护历史队列
            self.ambient_history.append(risk)
            if len(self.ambient_history) > self.history_maxlen:
                self.ambient_history.pop(0)

            avg_risk = sum(self.ambient_history) / len(self.ambient_history)
            if avg_risk > 0.6:
                await self.publish("acoustic.ambient_alert", {
                    "risk_score": avg_risk,
                    "reason": "持续检测到异常环境声音",
                    "history": self.ambient_history.copy()
                })

    async def handle_audio_ready(self, payload: Dict[str, Any]):
        """
        响应总线上的 audio.ready 指令，进行指令驱动的检测，
        并将环境上下文一并附上。
        """
        audio_path = payload.get("audio_path")
        if not audio_path:
            raise ValueError("audio_path missing in payload")

        # 加载音频文件（根据实际音频格式调整）
        audio = self._load_audio(audio_path)  # 返回 numpy array (16kHz)

        result = self.detector.detect(audio)
        # 注入环境风险（最近一次历史）
        ambient_risk = self.ambient_history[-1] if self.ambient_history else 0.0
        result["ambient_risk"] = ambient_risk
        result["is_increasing"] = self._is_risk_increasing()

        # 计算并添加置信度
        confidence = self._calc_confidence(result)
        result["confidence"] = confidence

        await self.publish("risk.acoustic", result)

    def _load_audio(self, path: str) -> np.ndarray:
        """加载音频文件，转换为 16kHz 单声道 numpy 数组"""
        import soundfile as sf
        audio, sr = sf.read(path)
        if sr != 16000:
            # 需要重采样，这里简化，实际可用 librosa
            raise NotImplementedError("Resampling not implemented")
        return audio

    def _is_risk_increasing(self) -> bool:
        """判断环境风险是否呈上升趋势"""
        if len(self.ambient_history) < 2:
            return False
        return self.ambient_history[-1] > self.ambient_history[-2]

    def _calc_confidence(self, result: Dict[str, Any]) -> float:
        """
        启发式置信度计算：
        - 基于 LA 和 PA 风险差异；差异越大，置信度越低。
        - 结合环境风险趋势。
        """
        la_risk = result.get("evidence", {}).get("la_risk", 0.0)
        pa_risk = result.get("evidence", {}).get("pa_risk", 0.0)
        diff = abs(la_risk - pa_risk)

        # 基础置信度：差异小于0.2时高置信，否则降低
        base = 1.0 - min(diff * 1.5, 0.8)
        # 环境趋势惩罚：若风险正在上升，置信度降低
        trend_penalty = 0.2 if self._is_risk_increasing() else 0.0
        return max(0.0, min(1.0, base - trend_penalty))
