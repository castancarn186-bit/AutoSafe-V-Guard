#风险融合决策引擎	收集 A/B/C 的报告，执行加权算法，输出最终决策。
import asyncio
import time
import random
from typing import List, Tuple
from core.protocol import RiskReport, DecisionType, AttackType
from core.state import shared_state
from hardware.gpio_ctrl import vguard_hw
from data.database_manager import db_manager


class VGuardEngine:
    def __init__(self):
        # 权重分配：体现安全策略重心 [cite: 12]
        # A(物理声学) 40%, B(ASR行为) 30%, C(语义校验) 30%
        self.weights = {"A": 0.4, "B": 0.3, "C": 0.3}

    async def run_pipeline(self):
        """
        核心调度流水线：并发调用模块 -> 风险融合 -> 物理执行
        """
        start_time = time.perf_counter()

        # 1. 并发调用 A/B/C 模块 (并行计算提升响应速度)
        # 在答辩时可以强调：系统总延迟 $Latency = \max(t_a, t_b, t_c)$
        tasks = [
            self._get_module_report("A"),
            self._get_module_report("B"),
            self._get_module_report("C")
        ]
        reports: List[RiskReport] = await asyncio.gather(*tasks)

        # 2. 更新性能遥测状态 (用于 UI 显示各模块耗时)
        for r in reports:
            shared_state.latencies[r.module_id] = r.latency_ms

        # 3. 风险融合算法 [cite: 14]
        total_risk, final_decision = self._fusion_logic(reports)

        # 4. 人机共驾覆盖逻辑 (一等奖加分项：功能安全保障)
        if shared_state.is_human_override:
            final_decision = DecisionType.PASS.value
            total_risk = total_risk * 0.5  # 人为干预时降低风险分显示

        # 5. 更新全局状态机
        shared_state.latest_reports = reports
        shared_state.total_risk = total_risk
        shared_state.decision = final_decision

        # 6. 物理执行：调用硬件接口 [cite: 11]
        vguard_hw.set_status(final_decision)

        # 7. 数据存证：将拦截行为计入影子模式数据库 [cite: 11]
        if final_decision != DecisionType.PASS.value or total_risk > 0.3:
            try:
                db_manager.save_log(total_risk, final_decision, reports)
            except Exception as e:
                print(f"[Engine] 存证失败: {e}")

    async def _get_module_report(self, module_id: str) -> RiskReport:
        """
        模拟模块检测逻辑，集成了攻击注入演示后门
        """
        start_t = time.perf_counter()

        # 模拟不同模块的基础计算开销
        delay = {"A": 0.01, "B": 0.02, "C": 0.01}
        await asyncio.sleep(delay.get(module_id, 0.01) + random.uniform(0, 0.01))

        risk = random.uniform(0.05, 0.15)
        reason = "监测正常"

        # --- 攻击模拟后门：用于演示现场一键触发 ---
        active_attack = shared_state.active_attack

        if module_id == "A" and active_attack == AttackType.REPLAY.value:
            risk = 0.95
            reason = "警告：检测到高频重放攻击特征"
        elif module_id == "B" and active_attack == AttackType.ADVERSARIAL.value:
            risk = 0.85
            reason = "异常：ASR 置信度发生非线性坍塌"
        elif module_id == "C":
            # 逻辑校验：高速行驶拦截危险指令
            if shared_state.vehicle_speed > 80 and (
                    "打开" in shared_state.asr_text or "后备箱" in shared_state.asr_text):
                risk = 0.98
                reason = f"危险：当前车速 {shared_state.vehicle_speed}km/h，已拦截车门操作"
            elif shared_state.weather == "暴雨" and "关闭" in shared_state.asr_text and "雨刮" in shared_state.asr_text:
                risk = 0.90
                reason = "警告：极端天气下禁止关闭安全辅助设备"

        latency = (time.perf_counter() - start_t) * 1000

        return RiskReport(
            module_id=module_id,
            risk_score=risk,
            suggestion=DecisionType.BLOCK.value if risk > 0.8 else DecisionType.PASS.value,
            reason=reason,
            evidence={"latency": latency},
            latency_ms=round(latency, 2)
        )

    def _fusion_logic(self, reports: List[RiskReport]) -> Tuple[float, str]:
        """
        多源风险融合算法 [cite: 14]
        """
        total_score = 0.0

        # 策略：一票否决 + 加权平均 [cite: 9]
        for r in reports:
            if r.risk_score > 0.9:
                return r.risk_score, DecisionType.BLOCK.value

            w = self.weights.get(r.module_id, 0.3)
            total_score += r.risk_score * w

        # 判定决策 [cite: 15]
        if total_score > 0.7:
            decision = DecisionType.BLOCK.value
        elif total_score > 0.4:
            decision = DecisionType.WARN.value
        else:
            decision = DecisionType.PASS.value

        return round(total_score, 2), decision
'''
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
        self.weights = {"A": 0.4, "B": 0.3, "C": 0.3}

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