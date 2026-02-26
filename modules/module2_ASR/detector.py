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
            self.model = ASRRiskModel()
            self.logger.info("ASR 风险评估模型加载成功")
        except Exception as e:
            self.logger.error(f"ASR 模型初始化失败: {e}")
            self.model = None

    def detect(self, data, context=None) -> DetectionResult:
        start_time = time.perf_counter()
        if self.model is None:
            return DetectionResult("B", 0.1, "PASS", "模型未初始化", 0.0)

        try:
            # ==========================================
            # 🐞 架构师修复：自适应输入数据类型
            # 如果是字符串路径，就加载它；如果是矩阵，直接用
            # ==========================================
            if isinstance(data, str):
                audio_matrix, _ = librosa.load(data, sr=16000)
            else:
                audio_matrix = data

            # 调用成员 B 原本的方法
            raw_res = self.model.compute_risk(audio_matrix)

            # ==========================================
            # 🐞 架构师修复：兼容对象与字典
            # 防止他返回的是 DataClass 而不是 dict 导致的 .get() 崩溃
            # ==========================================
            risk_score=0.0
            if isinstance(raw_res, dict):
                risk_score = raw_res.get("asr_risk_score", raw_res.get("risk", raw_res.get("score", 0.0)))
                text = raw_res.get("text", "")
            else:
                # 如果是个对象，用 getattr 取值
                risk_score = getattr(raw_res, "asr_risk_score", getattr(raw_res, "risk", getattr(raw_res, "score", 0.0)))
                text = getattr(raw_res, "text", "")

            return DetectionResult(
                module_id=self.module_id,
                risk_score=float(risk_score),
                decision="PASS" if risk_score < 0.4 else "REVIEW",
                reason=f"ASR分析完成",
                latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
                metadata={"text": text}
            )
        except Exception as e:
            self.logger.error(f"ASR 检测执行崩溃: {e}")
            return DetectionResult(self.module_id, 1.0, "BLOCK", f"ASR 模块故障: {str(e)}")