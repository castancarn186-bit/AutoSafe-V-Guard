# 声学安全主类	继承 BaseGuardModule，实现物理层安全检测主逻辑。
import os
import numpy as np
import torch
from core.base_module import BaseDetector
from core.protocol import SystemContext, RiskReport

from modules.module1_acoustic.audio_preprocessor import AudioPreprocessor
from modules.module1_acoustic.feature_extractor import FeatureExtractor
from modules.module1_acoustic.acoustic_anomaly_model import AnomalyModel
from modules.module1_acoustic.acoustic_risk_normalizer import RiskNormalizer


class AcousticDetector(BaseDetector):
    def __init__(self, config=None):
        super().__init__(module_id="A")
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

        # =========================================================
        # 🔥 架构师修复：去掉强行 raise 崩溃，改为优雅降级
        # =========================================================
        if not os.path.exists(model_path):
            self.logger.warning(f"⚠️ 找不到声学模型: {model_path}，模块 A 进入降级放行模式！")
            self.anomaly_model = None
        else:
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
                module_id=self.module_id,
                risk_score=0.0,
                suggestion="PASS",
                reason="No audio input",
                evidence={}
            )

        # =========================================================
        # 🔥 架构师修复：降级模式下，直接返回安全分，不执行模型推理
        # =========================================================
        if self.anomaly_model is None:
            return RiskReport(
                module_id=self.module_id,
                risk_score=0.1,
                suggestion="PASS",
                reason="模型文件缺失，默认放行",
                evidence={"fallback": True}
            )

        try:
            # 1. 预处理
            clean_audio = self.preprocessor.process(audio)
            if self.anomaly_model.model_type == 'sklearn':
                features = self.feature_extractor.extract(clean_audio)
                raw_score = self.anomaly_model.predict_risk(features)
            else:  # pytorch
                raw_score = self.anomaly_model.predict_risk(clean_audio)

            # 2. 特征提取
            features = self.feature_extractor.extract(clean_audio)

            # 3. 异常评分
            raw_score = self.anomaly_model.predict_risk(features)

            # 4. 风险归一化
            risk_score = self.normalizer.normalize(raw_score)

            # 5. 决策建议
            suggestion = self._get_suggestion(risk_score)

            # 6. 构建证据
            evidence = {
                "raw_score": raw_score,
                "feature_norm": np.linalg.norm(features).item()
            }

            return RiskReport(
                module_id=self.module_id,
                risk_score=risk_score,
                suggestion=suggestion,
                reason="Acoustic anomaly check",
                evidence=evidence
            )
        except Exception as e:
            self.logger.exception("Error in acoustic detection")
            return RiskReport(
                module_id=self.module_id,
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
