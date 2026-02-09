import flet as ft
import asyncio
from datetime import datetime
# 【关键修改】引入共享状态机
from core.state import shared_state

async def main_ui(page: ft.Page):
    # 1. 页面基本配置
    page.title = "V-Guard 安全监控仪表盘"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#0f172a"
    page.window_width = 1000
    page.window_height = 800
    page.padding = 30

    # 2. 定义界面元素
    title = ft.Text("V-GUARD SYSTEM MONITOR", size=32, weight="bold", color="#38bdf8")

    # 总风险圆环
    total_risk_chart = ft.ProgressRing(width=200, height=200, stroke_width=15, value=0, color="green")
    total_risk_text = ft.Text("0%", size=40, weight="bold")

    # A/B/C 模块卡片创建函数
    def create_module_card(name, color):
        return ft.Container(
            content=ft.Column([
                ft.Text(name, size=16, weight="bold"),
                ft.ProgressBar(width=250, value=0, color=color),
                ft.Text("Risk Score: 0.00", size=14, italic=True)
            ]),
            padding=20,
            bgcolor="#1e293b",
            border_radius=15,
            border=ft.Border.all(1, "#334155")
        )

    card_a = create_module_card("模块 A: 声学物理层", "#fbbf24")
    card_b = create_module_card("模块 B: ASR 行为安全", "#818cf8")
    card_c = create_module_card("模块 C: 语义校验层", "#34d399")

    log_box = ft.ListView(expand=True, spacing=10, padding=10)

    # 3. 布局组装
    page.add(
        title,
        ft.Divider(height=20, color="transparent"),
        ft.Row([
            ft.Stack([
                total_risk_chart,
                ft.Container(
                    total_risk_text,
                    alignment=ft.Alignment(0, 0),
                    width=200,
                    height=200
                )
            ]),
            ft.Column([card_a, card_b, card_c], spacing=20)
        ], alignment=ft.MainAxisAlignment.SPACE_EVENLY),
        ft.Divider(height=40),
        ft.Text("系统实时决策日志", size=18, weight="bold"),
        ft.Container(log_box, bgcolor="#020617", border_radius=10, height=200)
    )

    # 4. 【核心重构】动态更新逻辑：从大脑获取真实数据
    while True:
        # 从共享盒子读取数据
        total = shared_state.total_risk
        decision = shared_state.decision
        reports = shared_state.latest_reports

        # 如果后台还没产生数据，就等等
        if not reports:
            await asyncio.sleep(0.5)
            continue

        # 更新总风险 UI
        total_risk_chart.value = total
        total_risk_chart.color = "red" if total > 0.6 else "yellow" if total > 0.3 else "green"
        total_risk_text.value = f"{int(total * 100)}%"

        # 更新三个子模块的进度条和文本
        for r in reports:
            if r.module_id == "A":
                card_a.content.controls[1].value = r.risk_score
                card_a.content.controls[2].value = f"Risk Score: {r.risk_score:.2f}"
            elif r.module_id == "B":
                card_b.content.controls[1].value = r.risk_score
                card_b.content.controls[2].value = f"Risk Score: {r.risk_score:.2f}"
            elif r.module_id == "C":
                card_c.content.controls[1].value = r.risk_score
                card_c.content.controls[2].value = f"Risk Score: {r.risk_score:.2f}"

        # 插入最新决策日志
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_box.controls.insert(0, ft.Text(
            f"[{timestamp}] 决策结果: {decision} | 综合风险: {total:.2f}",
            color="#38bdf8" if decision == "PASS (放行)" else "#f87171"
        ))

        # 保持日志长度，防止内存溢出
        if len(log_box.controls) > 15:
            log_box.controls.pop()

        page.update()
        # 刷新频率建议为 1 秒，既能保证实时性，又不会太占树莓派资源
        await asyncio.sleep(1)

