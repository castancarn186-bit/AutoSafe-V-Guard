# core/state.py

class SystemState:
    """
    全系统共享状态机 (升级版)
    负责连接后台引擎、环境模拟器与前端 UI
    """

    def __init__(self):
        # --- 1. 基础控制 ---
        self.is_running = True

        # --- 2. 风险决策数据 (给 UI 的圆环和卡片用) ---
        self.total_risk = 0.0
        self.decision = "INITIALIZING"
        self.latest_reports = []

        # --- 3. 【新增】实时感知数据 (展示在右侧数据流看板) ---
        # 对应 ASR 识别结果
        self.asr_text = "等待语音输入..."
        # 对应成员 C 转换后的意图
        self.intent_label = "无"

        # --- 4. 【新增】环境与车辆状态 (由你负责模拟) ---
        self.vehicle_speed = 0  # 实时车速
        self.weather = "晴天"  # 环境天气
        self.execution_result = "待机"  # 实际执行结果 (拦截/放行)

        # --- 5. 【新增】长尾指令扩展状态 ---
        self.wipers_on = False  # 雨刮器状态
        self.turn_signal = "OFF"  # 转向灯: OFF/LEFT/RIGHT
        self.brake_pressure = 0.0  # 刹车压力


# 实例化全局唯一变量
shared_state = SystemState()