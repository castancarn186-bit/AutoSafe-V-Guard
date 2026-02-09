import flet as ft
import asyncio
from datetime import datetime
from core.state import shared_state


# --- 0. 自定义颜色函数 ---
def get_alpha_color(color_hex, opacity):
    try:
        if not color_hex.startswith("#"): return color_hex
        alpha = int(opacity * 255)
        return f"#{alpha:02x}{color_hex.lstrip('#')}"
    except:
        return color_hex


async def main_ui(page: ft.Page):
    # 挂载后台任务
    from main import detection_engine_task
    page.run_task(detection_engine_task)

    # --- 1. 页面配置 ---
    page.title = "V-Guard Pro"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#000000"
    page.window_width = 1280
    page.window_height = 800
    page.padding = 30
    page.theme = ft.Theme(font_family="Arial")

    # --- 颜色常量 ---
    METAL_GRADIENT_1 = ["#1c1c1e", "#2c2c2e"]
    ACCENT_CYAN = "#40c4ff"
    ACCENT_RED = "#ff3b30"
    ACCENT_GREEN = "#34c759"
    ACCENT_AMBER = "#ff9500"
    TEXT_SUBTLE = "#8e8e93"

    # --- 2. 基础组件工厂 (用 Container 模拟一切) ---

    def create_metallic_container(content, padding=20, expand=False, height=None):
        return ft.Container(
            content=content,
            padding=padding,
            border_radius=24,
            height=height,
            gradient=ft.LinearGradient(
                begin=ft.Alignment(-1, -1),
                end=ft.Alignment(1, 1),
                colors=METAL_GRADIENT_1
            ),
            border=ft.Border.all(1, "#3a3a3c"),
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=20,
                color="#80000000",  # 手动透明度
                offset=ft.Offset(0, 10),
            ),
            expand=expand,
        )

    def create_module_card(num, title, color_str):
        return create_metallic_container(
            content=ft.Row([
                # 左侧图标
                ft.Container(
                    content=ft.Text(num, size=18, weight="bold", color="#1c1c1e"),
                    alignment=ft.Alignment(0, 0),
                    bgcolor=color_str,
                    width=40, height=40, border_radius=12
                ),
                # 中间信息
                ft.Column([
                    ft.Text(title, size=16, weight="w600"),
                    ft.ProgressBar(value=0, color=color_str, bgcolor="#1c1c1e", height=6, border_radius=3),
                ], expand=True, spacing=8),
                # 右侧状态
                ft.Column([
                    ft.Text("0.00", size=18, weight="bold", color=color_str),
                    ft.Text("待机中", size=12, color=TEXT_SUBTLE, text_align="right")
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.END, spacing=2)
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            padding=ft.padding.symmetric(horizontal=20, vertical=18),
            expand=True
        )

    # --- 3. 核心组件 ---

    risk_ring = ft.ProgressRing(width=200, height=200, stroke_width=18, value=0, color=ACCENT_CYAN, bgcolor="#1c1c1e")
    risk_text = ft.Text("0%", size=64, weight="900")

    decision_tag = ft.Container(
        content=ft.Row([
            ft.Icon(ft.Icons.CIRCLE, size=12, color=ACCENT_GREEN),
            ft.Text("SYSTEM READY", size=14, weight="w600")
        ], spacing=8, tight=True),
        padding=ft.padding.symmetric(8, 16),
        border_radius=30,
        bgcolor="#2c2c2e",
        border=ft.Border.all(1, "#3a3a3c")
    )

    dashboard_container = create_metallic_container(
        content=ft.Column([
            ft.Text("综合风险指数", color=TEXT_SUBTLE, size=14),
            ft.Stack([
                risk_ring,
                ft.Container(
                    ft.Column([risk_text, decision_tag], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10),
                    alignment=ft.Alignment(0, 0),
                    width=180, height=180
                )
            ], alignment=ft.Alignment(0, 0)),
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=20),
        expand=True
    )

    card_a = create_module_card("A", "声学物理层探测", ACCENT_AMBER)
    card_b = create_module_card("B", "ASR 模型行为分析", ACCENT_CYAN)
    card_c = create_module_card("C", "语义与状态校验", ACCENT_GREEN)

    modules_container = ft.Column([card_a, card_b, card_c], expand=True, spacing=15)

    log_list = ft.ListView(expand=True, spacing=8, auto_scroll=True)
    log_container = create_metallic_container(
        content=ft.Column([
            ft.Row([ft.Icon(ft.Icons.TERMINAL, size=16, color=TEXT_SUBTLE),
                    ft.Text("实时安全决策日志流", color=TEXT_SUBTLE, weight="w600")], spacing=8),
            ft.Container(height=10),  # 手动 Spacer
            log_list
        ]),
        height=200
    )

    # --- 4. 布局 ---
    page.add(
        # 顶部栏
        ft.Row([
            ft.Icon(ft.Icons.SHIELD_MOON, color=ACCENT_CYAN, size=28),
            ft.Text("V-GUARD PRO", size=22, weight="900", italic=True),
            ft.Container(
                content=ft.Text("车载级防御系统激活", size=12, color=ACCENT_CYAN, weight="bold"),
                padding=ft.padding.symmetric(4, 10),
                border_radius=20,
                bgcolor=get_alpha_color(ACCENT_CYAN, 0.1)
            ),
            # 【修复点】用 expand=True 的 Container 代替 Spacer()
            ft.Container(expand=True),
            ft.Icon(ft.Icons.WIFI, color=TEXT_SUBTLE, size=18),
            ft.Icon(ft.Icons.BATTERY_FULL, color=TEXT_SUBTLE, size=18),
            ft.Text(datetime.now().strftime("%H:%M"), color=TEXT_SUBTLE, weight="bold")
        ], alignment=ft.MainAxisAlignment.CENTER),

        ft.Container(height=30),  # 手动 Spacer

        # 核心区
        ft.Row([
            dashboard_container,
            # 【修复点】用 Container 手写分割线 代替 VerticalDivider
            ft.Container(width=1, bgcolor="#2c2c2e", height=300, margin=ft.margin.symmetric(horizontal=20)),
            modules_container,
        ], expand=True),

        ft.Container(height=20),  # 手动 Spacer
        log_container
    )

    # --- 5. 数据循环 ---
    while True:
        total = shared_state.total_risk
        reports = shared_state.latest_reports
        decision = shared_state.decision

        color_state = ACCENT_RED if total > 0.6 else ACCENT_AMBER if total > 0.3 else ACCENT_GREEN

        risk_ring.value = total
        risk_ring.color = color_state
        risk_text.value = f"{int(total * 100)}%"

        decision_tag.content.controls[0].color = color_state
        decision_tag.content.controls[1].value = decision
        decision_tag.bgcolor = get_alpha_color(color_state, 0.2)

        if reports:
            for r in reports:
                target = card_a if r.module_id == "A" else card_b if r.module_id == "B" else card_c
                card_color = ACCENT_RED if r.risk_score > 0.6 else ACCENT_AMBER if r.risk_score > 0.3 else ACCENT_GREEN

                target.content.controls[2].controls[0].value = f"{r.risk_score:.2f}"
                target.content.controls[2].controls[0].color = card_color
                target.content.controls[1].controls[1].value = r.risk_score
                target.content.controls[1].controls[1].color = card_color
                reason = r.evidence.get("reason", "系统运行正常")
                target.content.controls[2].controls[1].value = reason

        timestamp = datetime.now().strftime("%H:%M:%S")
        log_icon = ft.Icons.CHECK_CIRCLE if total < 0.3 else ft.Icons.WARNING_AMBER if total < 0.6 else ft.Icons.BLOCK

        log_list.controls.insert(0, ft.Container(
            content=ft.Row([
                ft.Icon(log_icon, color=color_state, size=16),
                ft.Text(timestamp, color=TEXT_SUBTLE, size=13),
                ft.Text(decision, weight="bold", color="white"),
                ft.Text(f"综合风险值: {total:.2f}", color=color_state, italic=True, size=13)
            ], spacing=10),
            padding=ft.padding.symmetric(vertical=4),
            border=ft.border.only(bottom=ft.BorderSide(1, "#2c2c2e"))
        ))
        if len(log_list.controls) > 15: log_list.controls.pop()

        page.update()
        await asyncio.sleep(1)