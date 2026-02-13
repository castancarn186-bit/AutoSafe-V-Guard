# ui/components/history_panel.py
import flet as ft
import sqlite3
import json
from datetime import datetime


class HistoryPanel(ft.UserControl):
    def __init__(self):
        super().__init__()
        self.db_path = "data/defense_logs.db"

    def get_logs(self):
        """从数据库读取最近 10 条拦截记录"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT timestamp, total_risk, decision, reports_json FROM defense_logs ORDER BY timestamp DESC LIMIT 10")
            rows = cursor.fetchall()
            conn.close()
            return rows
        except:
            return []

    def build(self):
        logs = self.get_logs()

        # 创建数据表格
        table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("时间", size=12, color=ft.colors.WHITE70)),
                ft.DataColumn(ft.Text("分值", size=12, color=ft.colors.WHITE70)),
                ft.DataColumn(ft.Text("拦截原因", size=12, color=ft.colors.WHITE70)),
            ],
            rows=[
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(datetime.fromtimestamp(log[0]).strftime('%H:%M:%S'), size=11)),
                        ft.DataCell(
                            ft.Text(f"{log[1]:.2f}", color=ft.colors.RED_400 if log[1] > 0.7 else ft.colors.AMBER)),
                        ft.DataCell(
                            ft.Text(json.loads(log[3])[2]['reason'], size=11, overflow=ft.TextOverflow.ELLIPSIS)),
                    ]
                ) for log in logs
            ],
            heading_row_height=35,
            data_row_min_height=35,
            column_spacing=20,
        )

        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.icons.HISTORY, size=18, color=ft.colors.CYAN),
                    ft.Text("历史安全记录 (影子模式)", size=14, weight="bold"),
                ]),
                ft.Divider(height=1, color=ft.colors.WHITE10),
                ft.Container(content=table, padding=ft.padding.only(top=10))
            ]),
            padding=20,
            bgcolor=ft.colors.with_opacity(0.05, ft.colors.BLACK),
            border=ft.border.all(1, ft.colors.WHITE10),
            border_radius=12,
        )