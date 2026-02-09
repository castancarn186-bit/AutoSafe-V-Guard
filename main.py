import asyncio
import flet as ft
from core.engine import VGuardEngine
from core.state import shared_state
from ui.app import main_ui


async def detection_engine_task():
    """后台检测逻辑任务"""
    engine = VGuardEngine()
    print("后台逻辑引擎已就绪...")

    while shared_state.is_running:
        # 1. 模拟/获取 A/B/C 报告
        reports = engine.generate_mock_reports()

        # 2. 计算综合风险
        total_risk, decision = await engine.run_fusion(reports)

        # 3. 【关键】同步到共享状态机
        shared_state.latest_reports = reports
        shared_state.total_risk = total_risk
        shared_state.decision = decision

        await asyncio.sleep(1)  # 每秒检测一次


async def main():
    # 使用并发任务：同时启动检测引擎和 UI 界面
    # 注意：flet 的 run 逻辑稍后通过 main_ui 启动
    await asyncio.gather(
        detection_engine_task(),
        ft.app_async(target=main_ui)  # 启动异步 UI
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        shared_state.is_running = False