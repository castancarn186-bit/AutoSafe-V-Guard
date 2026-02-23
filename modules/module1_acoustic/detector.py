#声学安全主类	继承 BaseGuardModule，实现物理层安全检测主逻辑。
import os
import numpy as np
from core.base_module import BaseDetector
from core.protocol import SystemContext, RiskReport
from audio_preprocessor import AudioPreprocessor
from feature_extractor import FeatureExtractor
from acoustic_anomaly_model import AnomalyModel
from acoustic_risk_normalizer import RiskNormalizer

class AcousticDetector(BaseDetector):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.preprocessor = None
        self.feature_extractor = None
        self.anomaly_model = None
        self.normalizer = None

    def setup(self):
        """初始化所有子模块，加载模型"""
        # 预处理配置
        target_sr = self.config.get('target_sr', 16000)
        vad_aggressiveness = self.config.get('vad_aggressiveness', 2)
        self.preprocessor = AudioPreprocessor(
            target_sr=target_sr,
            vad_aggressiveness=vad_aggressiveness
        )

        # 特征提取器
        self.feature_extractor = FeatureExtractor(sr=target_sr)

        # 加载异常检测模型
        model_path = self.config.get(
            'model_path',
            os.path.join(os.path.dirname(__file__), 'models', 'ocsvm.pkl')
        )
        if not os.path.exists(model_path):
            self.logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Acoustic model missing: {model_path}")
        self.anomaly_model = AnomalyModel(model_path)

        # 风险归一化器
        norm_method = self.config.get('norm_method', 'sigmoid')
        self.normalizer = RiskNormalizer(method=norm_method)

        self.logger.info("AcousticDetector setup complete")

    def detect(self, ctx: SystemContext) -> RiskReport:
        """
        核心检测逻辑
        ctx.audio_frame: numpy array, 原始音频块
        """
        audio = ctx.audio_frame
        if audio is None or len(audio) == 0:
            return RiskReport(
                risk_score=0.0,
                suggestion="PASS",
                reason="No audio input",
                evidence={}
            )

        try:
            # 1. 预处理（假设输入音频已经是16kHz，但如果ctx中有原始采样率信息，可以传入）
            # 目前ctx可能不包含采样率，假设与目标一致；若不一致需在配置中说明
            clean_audio = self.preprocessor.process(audio)

            # 2. 特征提取
            features = self.feature_extractor.extract(clean_audio)

            # 3. 异常评分
            raw_score = self.anomaly_model.predict_risk(features)

            # 4. 风险归一化
            risk_score = self.normalizer.normalize(raw_score)

            # 5. 决策建议
            suggestion = self._get_suggestion(risk_score)

            # 6. 构建证据（可选，可用于调试）
            evidence = {
                "raw_score": raw_score,
                "feature_norm": np.linalg.norm(features).item()
            }

            return RiskReport(
                risk_score=risk_score,
                suggestion=suggestion,
                reason="Acoustic anomaly check",
                evidence=evidence
            )
        except Exception as e:
            self.logger.exception("Error in acoustic detection")
            return RiskReport(
                risk_score=0.5,  # 保守值
                suggestion="PASS",
                reason=f"Acoustic detection error: {str(e)}",
                evidence={}
            )

    def _get_suggestion(self, risk_score):
        """根据风险分数给出建议"""
        if risk_score < 0.3:
            return "PASS"
        elif risk_score < 0.6:
            return "CONFIRM"
        else:
            return "REJECT"
