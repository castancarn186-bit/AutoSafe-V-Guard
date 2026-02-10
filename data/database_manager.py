# data/database_manager.py
import sqlite3
import json
import time
import os


class DatabaseManager:
    def __init__(self, db_path="data/defense_logs.db"):
        self.db_path = db_path
        # 确保目录存在
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """初始化数据库表结构"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS defense_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    total_risk REAL,
                    decision TEXT,
                    reports_json TEXT
                )
            ''')
            conn.commit()

    def save_log(self, total_risk: float, decision: str, reports: list):
        """记录一次拦截或高风险行为"""
        # 将报告对象转为字典列表，再序列化为 JSON
        reports_data = [r.to_dict() for r in reports]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO defense_logs (timestamp, total_risk, decision, reports_json) VALUES (?, ?, ?, ?)",
                (time.time(), total_risk, decision, json.dumps(reports_data))
            )
            conn.commit()


# 单例，方便调用
db_manager = DatabaseManager()