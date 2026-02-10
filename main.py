import flet as ft
from ui.app import main_ui

if __name__ == "__main__":
    # 唯一入口，不要在这里写任何逻辑，全部交给 main_ui
    ft.app(target=main_ui)
'''
import flet as ft
import asyncio
from core.engine import VGuardEngine
from core.state import shared_state
from ui.app import main_ui


async def detection_engine_task():
    """后台逻辑引擎任务：生产者"""
    engine = VGuardEngine()
    print("[System]后台安全决策引擎激活")

    try:
        while shared_state.is_running:
            # 1. 模拟获取数据
            reports = engine.generate_mock_reports()
            # 2. 计算风险
            total_risk, decision = await engine.run_fusion(reports)
            # 3. 同步状态
            shared_state.latest_reports = reports
            shared_state.total_risk = total_risk
            shared_state.decision = decision

            await asyncio.sleep(1)
    except Exception as e:
        print(f"[Engine Error] {e}")


if __name__ == "__main__":
    ft.app(target=main_ui)
'''