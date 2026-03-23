# core/engine.py
import asyncio
import time
import logging
import librosa
from core.protocol import SystemContext, DetectionResult
from core.state import shared_state

# 导入探测器 (保持不变)
from modules.module1_acoustic.detector import AcousticDetector
from modules.module2_ASR.detector import ASRDetector
from modules.module3_semantic.detector import SemanticDetector

class VGuardEngine:
    def __init__(self):
        self.logger = logging.getLogger("VGuard.Engine")
        self.m1 = AcousticDetector(module_id='A')
        self.m2 = ASRDetector()
        self.m3 = SemanticDetector()
        self.m1.setup(); self.m2.setup(); self.m3.setup()
        self.weights = {"A": 0.30, "B": 0.2, "C": 0.5}

    async def analyze_risk_pipeline(self, audio_path: str, speed: float) -> dict:
        start_ts = time.perf_counter()
        try:
            audio_matrix, _ = librosa.load(audio_path, sr=16000, mono=True)
        except Exception as e:
            return {"total_risk": 1.0, "decision": "BLOCK", "asr_text": "音频加载失败"}

        # 🚀 改进：构建全量上下文，避免语义层因缺少字段报错
        ctx = SystemContext(
            audio_frame=audio_matrix,
            speed=float(speed),
            weather=shared_state.weather,
            has_pedestrians=False
        )

        # 1. 并发 A 和 B
        async def run_a(): return await asyncio.to_thread(self.m1.detect, ctx)
        async def run_b(): return await asyncio.to_thread(self.m2.detect, audio_matrix)

        results_ab = await asyncio.gather(run_a(), run_b(), return_exceptions=True)

        # 处理 A 模块
        report_a = results_ab[0] if not isinstance(results_ab[0], Exception) else \
                   DetectionResult("A", 0.5, "REVIEW", "声学层异常")

        # 🚀 处理 B 模块：修复“识别失败”和“0分”问题
        raw_b = results_ab[1]
        if isinstance(raw_b, Exception) or raw_b is None:
            report_b = DetectionResult("B", 0.5, "REVIEW", "ASR异常")
            extracted_text = "识别服务异常"
        elif isinstance(raw_b, dict):
            # 兼容 ASR 模型直接返回 dict 的情况
            extracted_text = raw_b.get('text', '识别为空')
            score = raw_b.get('risk_score', 0.0)
            report_b = DetectionResult("B", score, "PASS" if score < 0.6 else "BLOCK", "ASR分析完成")
            report_b.metadata = {"text": extracted_text}
        else:
            # 兼容 ASR 模型返回对象的情况
            report_b = raw_b
            extracted_text = getattr(report_b, "metadata", {}).get("text", "无文本")

        # 2. 串行执行 C 模块
        try:
            # 传入 ctx.__dict__ 以兼容 Pydantic 校验
            # 更新 ctx 的 asr_text 字段
            ctx.asr_text = extracted_text
            # 直接传递 ctx 对象
            report_c = await asyncio.to_thread(self.m3.run, ctx)
        except Exception as e:
            report_c = DetectionResult("C", 0.8, "BLOCK", f"语义层异常: {e}")

        # 3. 风险融合与决策
        reports = [report_a, report_b, report_c]
        total_risk = 0.0
        for r in reports:
            total_risk += r.risk_score * self.weights.get(r.module_id, 0.33)

        any_block = False
        any_review = False
        for r in reports:
            if r.decision == "BLOCK" or r.risk_score >= 0.8:
                any_block = True
                break
            if r.decision == "REVIEW" or r.risk_score >= 0.5:
                any_review = True

        if any_block:
            decision = "BLOCK"
        elif any_review:
            decision = "REVIEW"
        else:
            decision = "PASS"

        latency = round((time.perf_counter() - start_ts) * 1000, 2)

        self._print_terminal_log(extracted_text, speed, reports, total_risk, decision, latency)

        return {
            "total_risk": total_risk, "decision": decision,
            "reports": reports, "latency_ms": latency, "asr_text": extracted_text
        }

    def _print_terminal_log(self, text, speed, reports, total, decision, latency):
        # 统一使用对象属性访问，彻底修复 'RiskReport' object has no attribute 'get'
        print("\n" + "═" * 60)
        print(f"🛡️  [V-Guard 安全网关流水线] 评估完成 | 耗时: {latency}ms")
        print(f"🎤 识别文本: '{text}' | 🚗 车速: {speed} km/h")
        print("-" * 60)
        for r in reports:
            print(f"  ▶ [模块 {r.module_id}] 风险分: {r.risk_score:.3f} | 建议: {r.decision:<5} | 详情: {r.reason}")
        print("-" * 60)
        print(f"⚖️  最终网关指数: {total:.3f} -> 执行动作: 【{decision}】")
        print("═" * 60 + "\n")