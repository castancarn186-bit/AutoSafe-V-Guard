#风险融合决策引擎	收集 A/B/C 的报告，执行加权算法，输出最终决策。
# core/engine.py
import random
import time
from typing import List, Tuple
from core.protocol import RiskReport, DecisionType, SystemContext
from core.state import shared_state  # 引入共享状态，实现 UI 与引擎联动
from hardware.gpio_ctrl import vguard_hw  # 引入硬件控制（模拟器/真实树莓派）
from data.database_manager import db_manager  # 引入影子模式数据库


class VGuardEngine:
    def __init__(self):
        # 权重分配：A(声学) 30%, B(ASR) 30%, C(语义) 40%
        self.weights = {"A": 0.3, "B": 0.3, "C": 0.4}

    def generate_mock_reports(self) -> List[RiskReport]:
        """
        核心演示逻辑：基于 shared_state (车速、天气) 动态生成检测报告
        """
        speed = shared_state.vehicle_speed
        text = shared_state.asr_text
        weather = shared_state.weather

        # --- 模拟 第三部分：语义冲突检测 (C 模块) ---
        c_risk = 0.0
        c_reason = "逻辑合规"

        # 逻辑：高速拦截
        if speed > 80 and ("后备箱" in text or "门" in text):
            c_risk = 0.95
            c_reason = f"危险：高速({speed}km/h)禁止开启车门类设备"
        # 逻辑：恶劣天气拦截
        elif weather == "暴雨" and "关闭" in text and "雨刮" in text:
            c_risk = 0.85
            c_reason = "危险：极端天气下禁止关闭安全设备"
        else:
            c_risk = random.uniform(0.0, 0.1)

        # 模拟 第一部分：声学探测 (A 模块)
        report_a = RiskReport(
            module_id="A",
            risk_score=random.uniform(0.0, 0.2),
            suggestion=DecisionType.PASS.value,
            reason="声纹特征正常",
            evidence={"spectrogram": "normal"},
            latency_ms=random.uniform(5, 12)
        )

        # 模拟 第二部分：ASR 行为 (B 模块)
        report_b = RiskReport(
            module_id="B",
            risk_score=random.uniform(0.0, 0.2),
            suggestion=DecisionType.PASS.value,
            reason="识别置信度高",
            evidence={"confidence": 0.98},
            latency_ms=random.uniform(15, 25)
        )

        # 模拟 第三部分：最终生成 (C 模块)
        report_c = RiskReport(
            module_id="C",
            risk_score=c_risk,
            suggestion=DecisionType.BLOCK.value if c_risk > 0.8 else DecisionType.PASS.value,
            reason=c_reason,
            evidence={"policy_check": "failed" if c_risk > 0.5 else "passed"},
            latency_ms=random.uniform(5, 10)
        )

        return [report_a, report_b, report_c]

    async def run_fusion(self, reports: List[RiskReport]) -> Tuple[float, str]:
        """
        风险融合引擎：计算总分 -> 物理执行 -> 数据存证
        """
        if not reports:
            return 0.0, "System Idle"

        total_score = 0.0
        decision = DecisionType.PASS.value

        # 1. 逻辑判定：加权计算 + 一票否决
        for r in reports:
            if r.risk_score > 0.8:  # 触发一票否决
                total_score = r.risk_score
                decision = DecisionType.BLOCK.value
                break
            w = self.weights.get(r.module_id, 0.0)
            total_score += r.risk_score * w
        else:
            # 正常加权判定
            if total_score > 0.7:
                decision = DecisionType.BLOCK.value
            elif total_score > 0.4:
                decision = DecisionType.WARN.value
            else:
                decision = DecisionType.PASS.value

        # 2. 物理执行：调用硬件接口（亮灯/断开继电器）
        vguard_hw.set_status(decision)

        # 3. 数据存证：将高风险行为计入影子模式数据库
        if decision != DecisionType.PASS.value or total_score > 0.3:
            try:
                db_manager.save_log(total_score, decision, reports)
            except Exception as e:
                print(f"[Engine] 数据库存证失败: {e}")

        return round(total_score, 2), decision
'''
import asyncio
from typing import List
from core.protocol import RiskReport

class VGuardEngine:
    def __init__(self, modules: List):
        self.modules = modules
        # 风险权重分配 (这是答辩时可以展开讲的“算法优化点”)
        self.weights = {
            "A": 0.4,  # 声学物理层风险最高 (直接攻击)
            "B": 0.3,  # ASR 行为
            "C": 0.3   # 语义逻辑
        }

    async def run_pipeline(self, audio_data: bytes, vehicle_state: dict):
        """
        核心调度流水线
        """
        context = {"audio": audio_data, "state": vehicle_state}

        # 1. 并发调用 A/B/C 模块 (并行计算提升响应速度)
        # 就像三个专家同时对病人进行诊断
        tasks = [mod.detect(context) for mod in self.modules]
        reports: List[RiskReport] = await asyncio.gather(*tasks)

        # 2. 风险融合算法 (Weighted Sum Model)
        # 公式: $TotalRisk = \sum (Score_i \times Weight_i)$
        total_risk = sum(
            r.risk_score * self.weights.get(r.module_id, 0.1)
            for r in reports
        )

        # 3. 最终决策判决
        if total_risk > 0.7:
            decision = "BLOCK"
        elif total_risk > 0.4:
            decision = "WARN"
        else:
            decision = "PASS"

        return {
            "final_decision": decision,
            "total_risk": round(total_risk, 2),
            "breakdown": [r.to_dict() for r in reports]
        }
'''