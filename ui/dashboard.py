#可视化 App (Flet)	实时显示波形、各模块风险分、拦截历史
import flet as ft
import asyncio
from datetime import datetime
from core.state import shared_state


async def main_ui(page: ft.Page):
    page.title = "V-Guard Pro | 智能座舱语音安全防御系统"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#05070a"  # 极深色调
    page.padding = 20
    page.window_width = 1200
    page.window_height = 900

    # --- 样式定义 ---
    ACCENT_COLOR = "#00f2ff"  # 科技青
    DANGER_COLOR = "#ff4b2b"  # 警示红

    # --- UI 组件定义 ---
    # 顶部状态栏
    status_text = ft.Text("SYSTEM ACTIVE", color=ACCENT_COLOR, weight="bold")
    cpu_text = ft.Text("LATENCY: 45ms", size=12, color="#64748b")

    # 核心大仪表盘
    risk_ring = ft.ProgressRing(width=220, height=220, stroke_width=12, value=0, color=ACCENT_COLOR)
    risk_value_text = ft.Text("0%", size=48, weight="bold")
    decision_tag = ft.Container(
        content=ft.Text("INITIALIZING", size=12, weight="bold"),
        padding=ft.padding.symmetric(10, 20),
        border_radius=20,
        bgcolor="#1e293b"
    )

    # 证据链展示区
    def create_evidence_card(module_id, title, color):
        return ft.Container(
            expand=True,
            bgcolor="#0f172a",
            padding=20,
            border_radius=12,
            border=ft.Border.all(1, "#1e293b"),
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.icons.SHIELD_OUTLINED, color=color, size=20),
                    ft.Text(title, size=16, weight="bold"),
                ]),
                ft.ProgressBar(value=0, color=color, bgcolor="#1e293b", height=8),
                ft.Text("待检...", size=12, color="#94a3b8", italic=True),  # 存放证据文字
            ], spacing=10)
        )

    card_a = create_evidence_card("A", "物理声学安全", "#fbbf24")
    card_b = create_evidence_card("B", "ASR 模型行为", "#818cf8")
    card_c = create_evidence_card("C", "驾驶状态校验", "#34d399")

    # 实时日志区
    log_list = ft.ListView(expand=True, spacing=5, auto_scroll=True)

    # --- 布局构建 ---
    page.add(
        # Header
        ft.Row([
            ft.Text("V-GUARD", size=24, weight="bold", color="white"),
            ft.VerticalDivider(width=10),
            status_text,
            ft.Spacer(),
            cpu_text
        ]),
        ft.Divider(height=1, color="#1e293b"),

        # Main Content
        ft.Row([
            # 左侧核心监控
            ft.Container(
                expand=2,
                content=ft.Column([
                    ft.Text("实时风险评估评分", size=14, color="#64748b"),
                    ft.Stack([
                        ft.Container(risk_ring, alignment=ft.alignment.center),
                        ft.Container(
                            ft.Column([risk_value_text, decision_tag], horizontal_alignment="center", spacing=0),
                            alignment=ft.alignment.center
                        )
                    ], height=250),
                    ft.Divider(height=30, color="transparent"),
                    ft.Text("多源风险因子分布", size=14, color="#64748b"),
                    ft.Row([card_a, card_b, card_c], spacing=15)
                ], horizontal_alignment="center")
            ),
            # 右侧流水线与日志
            ft.Container(
                expand=1,
                bgcolor="#0a0f1e",
                border_radius=15,
                padding=20,
                content=ft.Column([
                    ft.Text("🛡️ 决策引擎日志", size=16, weight="bold"),
                    ft.Container(log_list, expand=True),
                    ft.Divider(color="#1e293b"),
                    ft.Text("答辩演示控制台", size=14, color="#64748b"),
                    ft.Row([
                        ft.ElevatedButton("正常指令", icon=ft.icons.PLAY_ARROW, on_click=lambda _: None),
                        ft.ElevatedButton("注入攻击", icon=ft.icons.BUG_REPORT, color="red", on_click=lambda _: None),
                    ], alignment="center")
                ])
            )
        ], expand=True)
    )

    # --- 数据驱动逻辑 ---
    async def refresh_loop():
        while True:
            # 1. 从共享状态读取
            total = shared_state.total_risk
            decision = shared_state.decision
            reports = shared_state.latest_reports

            if reports:
                # 2. 更新中央大仪表
                risk_ring.value = total
                risk_ring.color = DANGER_COLOR if total > 0.6 else ft.colors.AMBER if total > 0.3 else ACCENT_COLOR
                risk_value_text.value = f"{int(total * 100)}%"
                decision_tag.content.value = decision
                decision_tag.bgcolor = "#450a0a" if total > 0.6 else "#1e293b"

                # 3. 更新各模块证据链 (可解释性展示)
                for r in reports:
                    card = None
                    if r.module_id == "A":
                        card = card_a
                    elif r.module_id == "B":
                        card = card_b
                    elif r.module_id == "C":
                        card = card_c

                    if card:
                        # 更新进度条
                        card.content.controls[1].value = r.risk_score
                        # 更新证据文字（核心加分项：展示为什么这个分）
                        reason = r.evidence.get("reason", "检测结果正常")
                        card.content.controls[2].value = f"原因: {reason}"
                        card.content.controls[2].color = ft.colors.RED_400 if r.risk_score > 0.5 else "#94a3b8"

                # 4. 日志流
                log_list.controls.insert(0, ft.Text(
                    f"[{datetime.now().strftime('%H:%M:%S')}] {decision} (Score: {total:.2f})",
                    size=12, color="#cbd5e1"
                ))
                if len(log_list.controls) > 20: log_list.controls.pop()

            page.update()
            await asyncio.sleep(1)

    page.run_task(refresh_loop)

# 注意：此文件不包含 ft.app()，因为它将由 main.py 统一启动