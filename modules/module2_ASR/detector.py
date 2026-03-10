# modules/module2_ASR/detector.py
import time
import logging
import librosa
from core.base_module import BaseDetector, DetectionResult
from modules.module2_ASR.asr_risk_model import ASRRiskModel


class ASRDetector(BaseDetector):
    def __init__(self):
        super().__init__(module_id="B")
        self.logger = logging.getLogger("VGuard.ModuleB")
        try:
            # 💡 成员 B 的核心模型加载
            self.model = ASRRiskModel()
            self.logger.info("ASR 风险评估模型加载成功")
        except Exception as e:
            self.logger.error(f"ASR 模型初始化失败: {e}")
            self.model = None

    def detect(self, data, context=None) -> DetectionResult:
        """
        data: 可以是音频文件路径 (str) 或 numpy 矩阵 (ndarray) [cite: 191]
        """
        start_time = time.perf_counter()

        if self.model is None:
            return DetectionResult("B", 1.0, "BLOCK", "ASR模块未初始化", 0.0)

        try:
            # 1. 自动适配输入格式 [cite: 191, 192]
            if isinstance(data, str):
                audio_matrix, _ = librosa.load(data, sr=16000)
            else:
                audio_matrix = data

            # 2. 调用核心风险计算 [cite: 157, 163]
            # 该方法返回一个包含 text, confidence, risk_score 等字段的字典
            raw_res = self.model.compute_risk(audio_matrix)

            # 3. 提取分值与文本 [cite: 163, 193]
            # 成员 B 的代码返回 risk_score，范围 [0, 1]，0 表示安全 [cite: 160, 161]
            risk_score = raw_res.get("risk_score", 0.0)
            text = raw_res.get("text", "")

            # 4. 映射到系统决策逻辑 [cite: 161, 194]
            # 根据成员 B 的设定：<0.3 低风险，>0.7 高风险 [cite: 161, 162]
            decision = "PASS"
            if risk_score > 0.7:
                decision = "BLOCK"
            elif risk_score > 0.3:
                decision = "REVIEW"

            latency = (time.perf_counter() - start_time) * 1000

            # 🚀 返回全系统统一协议
            return DetectionResult(
                module_id=self.module_id,
                risk_score=float(risk_score),
                decision=decision,
                reason=f"ASR置信度评估完成: {raw_res.get('risk_level', '')}",
                latency_ms=round(latency, 2),
                metadata={
                    "text": text,
                    "confidence": raw_res.get("confidence", 0.0),
                    "timings": raw_res.get("timings", {})
                }
            )

        except Exception as e:
            self.logger.error(f"ASR 检测执行崩溃: {e}")
            return DetectionResult(self.module_id, 0.8, "REVIEW", f"ASR 内部错误: {str(e)}")

