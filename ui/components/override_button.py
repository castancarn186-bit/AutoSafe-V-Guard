import flet as ft
from core.state import shared_state


class EmergencyOverrideButton(ft.UserControl):
    def __init__(self):
        super().__init__()
        self.is_active = False

    def toggle_override(self, e):
        self.is_active = not self.is_active
        # 更新全局状态，告诉引擎：人类已接管，强制放行
        shared_state.is_human_override = self.is_active

        # 改变按钮视觉效果
        self.btn.content.controls[0].color = ft.colors.AMBER if self.is_active else ft.colors.WHITE
        self.btn.border = ft.border.all(2, ft.colors.AMBER if self.is_active else ft.colors.WHITE24)
        self.btn.update()

        print(f"[Human-in-the-loop] 紧急覆盖状态: {self.is_active}")

    def build(self):
        self.btn = ft.Container(
            content=ft.Row([
                ft.Icon(ft.icons.GPP_MAYBE, color=ft.colors.WHITE),
                ft.Text("HUMAN OVERRIDE", weight="bold", size=12)
            ], alignment=ft.MainAxisAlignment.CENTER),
            width=180,
            height=50,
            border=ft.border.all(1, ft.colors.WHITE24),
            border_radius=8,
            bgcolor=ft.colors.with_opacity(0.1, ft.colors.WHITE),
            on_click=self.toggle_override,
            animate=ft.animation.Animation(300, ft.AnimationCurve.DECELERATE),
        )
        return self.btn