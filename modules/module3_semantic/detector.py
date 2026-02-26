# modules/module3_semantic/detector.py
import time
import os
from core.base_module import BaseDetector, DetectionResult

# =========================================================
# 🏆 架构级规范修改：使用绝对路径导入成员 C 的算法库
# 直接精确到 src 文件夹，彻底抛弃 sys.path 补丁
# =========================================================
from modules.module3_semantic.semantic_lib import SemanticLibrary
from modules.module3_semantic.state_model import DrivingStateModel
from modules.module3_semantic.risk_assessment import RiskManager


class SemanticDetector(BaseDetector):
    def __init__(self):
        super().__init__(module_id="C")

        # 获取项目根目录，以便准确定位 semantic_model 文件夹
        # __file__ 是 detector.py 的路径 (modules/module3_semantic/detector.py)
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(ROOT_DIR, "modules", "semantic_model")

        # 1. 加载语义库
        self.lib = SemanticLibrary(model_name=model_path)

        # 2. 加载状态模型与风险管理器
        self.state_model = DrivingStateModel()
        self.risk_mgr = RiskManager(input_dim=16)  # 源码定义为 16 维

    def detect(self, data: str, context: dict = None) -> DetectionResult:
        """
        data: ASR 识别出的文本
        context: 驾驶上下文，例如 {'speed': 80, 'gear': 'D'}
        """
        start_time = time.perf_counter()

        try:
            # 1. 语义特征提取
            standard_cmd, _, _ = self.lib.match_intent(str(data))
            cmd_embedding = self.lib.model.encode([standard_cmd])[0]

            # 2. 驾驶状态向量化
            self.state_model.update_from_sensors(context or {})
            state_vector = self.state_model.get_state_vector()

            # 3. 风险融合评估
            input_tensor = self.risk_mgr.prepare_input(cmd_embedding, state_vector)
            risk_score = self.risk_mgr.evaluate(input_tensor)

            # 4. 判定决策 (DENY -> BLOCK, ALLOW -> PASS)
            raw_decision, _ = self.risk_mgr.generate_risk_matrix(risk_score)
            decision = "BLOCK" if raw_decision == "DENY" else "PASS"

            return DetectionResult(
                module_id=self.module_id,
                risk_score=float(risk_score),
                decision=decision,
                reason=f"意图: {standard_cmd} | 状态校验" if decision == "PASS" else f"非法意图: {standard_cmd}",
                latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
                metadata={"cmd": standard_cmd, "speed": context.get('speed', 0)}
            )
        except Exception as e:
            return DetectionResult(self.module_id, 1.0, "BLOCK", f"语义模块异常: {str(e)}")