import flet as ft
import asyncio
import os
import wave
import pyaudio
import threading
from datetime import datetime

# 核心架构导入
from core.state import shared_state, ModuleReport
from core.engine import VGuardEngine
from modules.module2_ASR.asr_risk_model import ASRRiskModel

# 强制修复多线程 DLL 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 样式常量 (保持原样) ---
BG_COLOR = "#080c11"
PANEL_BG = "#0d131a"
BORDER_COLOR = "#1a2636"
CYAN_GLOW = "#00f2ff"
RED_GLOW = "#ff2a2a"
ORANGE_GLOW = "#ffa500"
GREEN_GLOW = "#34c759"
TEXT_MUTED = "#647b91"


def get_border(width, color):
    side = ft.BorderSide(width, color)
    return ft.Border(top=side, bottom=side, left=side, right=side)


def create_glowing_panel(title, content, glow_color=CYAN_GLOW, expand=True):
    return ft.Container(
        expand=expand, padding=20, bgcolor=PANEL_BG, border_radius=15,
        border=get_border(1, BORDER_COLOR),
        shadow=ft.BoxShadow(spread_radius=0, blur_radius=10, color=f"{glow_color}40", offset=ft.Offset(0, 0)),
        content=ft.Column([
            ft.Text(title, size=14, weight="900", color="white"),
            ft.Divider(height=1, color=BORDER_COLOR),
            content
        ])
    )


# --- 录音辅助工具 (工程化封装) ---
def record_audio_blocking(filename="realtime_mic.wav", duration=3):
    """阻塞式录音，由 asyncio.to_thread 调用"""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    for _ in range(0, int(RATE / CHUNK * duration)):
        frames.append(stream.read(CHUNK))
    stream.stop_stream()
    stream.close()
    p.terminate()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    return filename


async def main_ui(page: ft.Page):
    # ==========================================
    # 0. 引擎与模型初始化
    # ==========================================
    engine = VGuardEngine()
    # 在 UI 层面持久化 B 模块模型，避免重复加载
    if not hasattr(shared_state, 'asr_model'):
        shared_state.asr_model = ASRRiskModel()

    async def detection_engine_task():
        print("[System] 后台安全决策引擎已就绪...")
        while shared_state.is_running:
            # 此处保持与原有逻辑兼容，处理非语音类的状态同步
            await asyncio.sleep(1)

    # ==========================================
    # 1. 页面基本配置
    # ==========================================
    page.title = "V-Guard Pro | 智能座舱防御总控"
    page.bgcolor = BG_COLOR
    page.padding = 20
    page.window_width = 1440
    page.window_height = 850
    page.theme_mode = ft.ThemeMode.DARK

    # ==========================================
    # 2. 交互控制事件
    # ==========================================
    async def start_voice_defense(e):
        """实时语音防御触发：录音 -> 分析 -> 渲染"""
        btn_voice.disabled = True
        btn_voice.text = "正在录制指令..."
        btn_voice.icon = ft.Icons.MIC_EXTERNAL_ON
        btn_voice.icon_color = RED_GLOW
        page.update()

        # 1. 录制音频 (建议改为5秒，10秒太久了)
        audio_file = await asyncio.to_thread(record_audio_blocking, "realtime_mic.wav", 5)

        btn_voice.text = "安全决策引擎分析中..."
        btn_voice.icon_color = ORANGE_GLOW
        page.update()

        # 2. 准备数据并推理
        import librosa
        try:
            audio_matrix, _ = librosa.load(audio_file, sr=16000)
            # 调用 B 成员的风险评估模型
            asr_res = await asyncio.to_thread(shared_state.asr_model.compute_risk, audio_matrix)
        except Exception as ex:
            print(f"❌ 推理过程发生错误: {ex}")
            btn_voice.disabled = False
            btn_voice.text = "开启语音防御监测"
            page.update()
            return

        # ================= [核心修改：全属性捕获逻辑] =================
        # 1. 定义默认值，防止下方代码标红
        raw_text = "识别失败"
        risk_b = 1.0
        conf_score = 0.0
        stab_score = 0.0

        # 2. 打印调试信息（这是找回文字的关键）
        print(f"🔍 [DEBUG] B模块返回对象所有属性: {dir(asr_res)}")

        # 3. 提取风险分数 (对应你标红的地方)
        risk_b = getattr(asr_res, "risk_score", 1.0)

        # 4. 提取文字 (多路径尝试)
        # 尝试顺序: text -> transcription -> result -> content -> asr_text
        for attr in ["text", "transcription", "result", "content", "asr_text"]:
            if hasattr(asr_res, attr):
                val = getattr(asr_res, attr)
                if val and len(str(val)) > 0:
                    raw_text = str(val)
                    break

        # 如果通过属性找不到，尝试转字典查找 (Pydantic特有方法)
        if raw_text == "识别失败" and hasattr(asr_res, "dict"):
            d = asr_res.dict()
            raw_text = d.get("text", d.get("transcription", "识别失败"))

        # 5. 提取置信度与稳定性
        conf_metrics = getattr(asr_res, "confidence_metrics", None)
        stab_metrics = getattr(asr_res, "stability_metrics", None)
        if conf_metrics:
            conf_score = getattr(conf_metrics, "confidence_score", 0.0)
        if stab_metrics:
            stab_score = getattr(stab_metrics, "stability_score", 0.0)
        # =============================================================

        # 3. 构造报告并更新 UI
        report_b = ModuleReport(
            module_id="B",
            risk_score=risk_b,
            status="PASS" if risk_b < 0.75 else "BLOCK",
            reason=f"置信度:{conf_score:.2f} | 稳定性:{stab_score:.2f}"
        )

        # 繁简转换并存入状态
        shared_state.asr_text = raw_text.replace("請", "请").replace("開", "开").replace("機", "机")

        # 模拟 A/C 模块数据
        report_a = ModuleReport(module_id="A", risk_score=0.05, status="SAFE", reason="物理层正常")
        report_c = ModuleReport(module_id="C", risk_score=0.10, status="SAFE", reason="语义符合场景")

        # 更新全局共享状态
        shared_state.latest_reports = [report_a, report_b, report_c]
        shared_state.total_risk = max(report_a.risk_score, report_b.risk_score, report_c.risk_score)
        shared_state.decision = "BLOCK" if shared_state.total_risk > 0.75 else "PASS"

        # 4. 恢复按钮并刷新页面
        btn_voice.disabled = False
        btn_voice.text = "开启语音防御监测"
        btn_voice.icon = ft.Icons.MIC
        btn_voice.icon_color = CYAN_GLOW
        page.update()

    def update_speed(e):
        val = int(e.control.value)
        shared_state.vehicle_speed = val
        speed_text.value = str(val)
        speed_ring.value = min(val / 200.0, 1.0)
        page.update()

    def update_weather(e):
        shared_state.weather = e.control.value
        weather_text.value = f"环境: {e.control.value}"
        page.update()

    def submit_asr(e):
        if asr_input.value:
            shared_state.asr_text = asr_input.value
            # 手动注入模拟一个 PASS 状态
            shared_state.decision = "PASS"
            page.update()

    def trigger_scenario(speed, weather, text):
        shared_state.vehicle_speed = speed
        shared_state.weather = weather
        shared_state.asr_text = text
        shared_state.decision = "BLOCK" if speed > 100 else "PASS"  # 简单的场景逻辑

        speed_slider.value = speed
        speed_text.value = str(speed)
        speed_ring.value = min(speed / 200.0, 1.0)
        weather_dropdown.value = weather
        weather_text.value = f"环境: {weather}"
        asr_input.value = text
        page.update()

    # ==========================================
    # 3. UI 控件搭建 (保持原有精美布局)
    # ==========================================

    # --- 左侧：环境与车控模拟 ---
    speed_text = ft.Text(str(shared_state.vehicle_speed), size=60, weight="900", color="white")
    speed_ring = ft.ProgressRing(width=250, height=250, stroke_width=15, value=0.0, color=CYAN_GLOW,
                                 bgcolor=BORDER_COLOR)
    weather_text = ft.Text(f"环境: {shared_state.weather}", size=14, color=GREEN_GLOW, weight="bold")

    speedometer = ft.Stack([
        speed_ring,
        ft.Container(
            content=ft.Column([
                speed_text, ft.Text("km/h", size=16, color=TEXT_MUTED, weight="bold"),
                ft.Text("D", size=24, color=CYAN_GLOW, weight="900"), weather_text
            ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=0),
            alignment=ft.Alignment(0, 0), width=250, height=250
        )
    ])

    speed_slider = ft.Slider(min=0, max=200, divisions=40, value=shared_state.vehicle_speed, active_color=CYAN_GLOW,
                             on_change=update_speed)
    weather_dropdown = ft.Dropdown(
        options=[ft.DropdownOption("晴朗"), ft.DropdownOption("多云"), ft.DropdownOption("暴雨"),
                 ft.DropdownOption("暴雪")],
        value=shared_state.weather, width=150, border_color=CYAN_GLOW, color="white", on_select=update_weather
    )

    panel_left = create_glowing_panel("CONTEXT SIMULATOR", ft.Column([
        ft.Container(height=10),
        ft.Container(speedometer, alignment=ft.Alignment(0, 0)),
        ft.Container(height=20),
        ft.Text("SPEED CONTROL (车速)", size=12, weight="bold", color="white"),
        speed_slider,
        ft.Row([ft.Text("WEATHER (天气):", size=12, weight="bold", color="white"), weather_dropdown],
               alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
    ]))

    # --- 中侧：核心防御管线 ---
    user_bubble_text = ft.Text("等待语音输入...", size=18, color="white")
    user_bubble = ft.Container(content=user_bubble_text, padding=15, border_radius=15, border=get_border(2, CYAN_GLOW),
                               bgcolor="#00f2ff10")

    system_icon = ft.Icon(ft.Icons.SHIELD, color=CYAN_GLOW, size=40)
    system_title = ft.Text("● 监控中 (MONITOR)", size=22, weight="900", color=CYAN_GLOW)
    system_reason = ft.Text("系统运行正常", size=16, color="white")

    system_bubble = ft.Container(
        content=ft.Row([system_icon, ft.Column([system_title, system_reason], spacing=2)]),
        padding=20, border_radius=15, border=get_border(2, CYAN_GLOW), bgcolor="#00f2ff10"
    )

    def xray_box(title, icon_name):
        val_text = ft.Text("Risk: 0.00", size=10, color=TEXT_MUTED)
        box = ft.Container(
            content=ft.Column([
                ft.Icon(icon_name, color=CYAN_GLOW, size=30),
                ft.Text(title, size=12, weight="bold", color="white"), val_text
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=15, border=get_border(1, CYAN_GLOW), border_radius=10, bgcolor=BORDER_COLOR
        )
        return box, val_text

    box_a, xray_a_val = xray_box("声学", ft.Icons.MULTILINE_CHART)
    box_b, xray_b_val = xray_box("行为", ft.Icons.SPEED)
    box_c, xray_c_val = xray_box("语义", ft.Icons.RULE)

    xray_row = ft.Container(
        content=ft.Column([
            ft.Text("VIEW探测器", size=12, weight="bold", color="white"),
            ft.Row([box_a, ft.Icon(ft.Icons.ARROW_FORWARD, color=CYAN_GLOW), box_b,
                    ft.Icon(ft.Icons.ARROW_FORWARD, color=CYAN_GLOW), box_c], alignment=ft.MainAxisAlignment.CENTER)
        ]), padding=15, border=get_border(1, BORDER_COLOR), border_radius=10
    )

    panel_mid = create_glowing_panel("V-GUARD DEFENSE PIPELINE", ft.Column([
        ft.Container(height=20), user_bubble, ft.Container(height=40),
        ft.Container(system_bubble, alignment=ft.Alignment(1, 0)), ft.Container(expand=True), xray_row
    ]))

    # --- 右侧：测试注入与预设场景 ---
    # 核心录音按钮
    btn_voice = ft.ElevatedButton(
        "开启语音防御监测", icon=ft.Icons.MIC,
        color=CYAN_GLOW, bgcolor=BG_COLOR, on_click=start_voice_defense,
        height=50
    )

    asr_input = ft.TextField(label="手动文字模拟注入", expand=True, border_color=CYAN_GLOW, color="white",
                             on_submit=submit_asr)
    btn_send = ft.IconButton(icon=ft.Icons.SEND, icon_color=CYAN_GLOW, on_click=submit_asr)

    def scenario_button(title, subtitle, color, speed, weather, text):
        return ft.Container(
            content=ft.Column(
                [ft.Text(title, weight="bold", color=color), ft.Text(subtitle, size=10, color=TEXT_MUTED)], spacing=2),
            padding=15, border=get_border(1, color), border_radius=8, bgcolor=f"{color}10",
            on_click=lambda _: trigger_scenario(speed, weather, text), ink=True
        )

    log_list = ft.ListView(expand=True, spacing=5, auto_scroll=True)

    panel_right = create_glowing_panel("ATTACK & TELEMETRY", ft.Column([
        ft.Text("实时语音", size=12, weight="bold", color="white"),
        ft.Container(btn_voice, alignment=ft.Alignment(0, 0)),
        ft.Divider(height=10, color=BORDER_COLOR),
        ft.Text("场景注入", size=12, weight="bold", color="white"),
        ft.Row([asr_input, btn_send]),
        scenario_button("🚗 高速越权危险", "120km/h + 开启后备箱", RED_GLOW, 120, "晴朗", "打开后备箱"),
        scenario_button("🌧️ 极端天气冲突", "40km/h + 暴雨 + 关雨刮", ORANGE_GLOW, 40, "暴雨", "关闭感应雨刮器"),
        scenario_button("🅿️ 停车合规放行", "0km/h + 开启后备箱", GREEN_GLOW, 0, "晴朗", "打开后备箱"),
        ft.Divider(height=10, color=BORDER_COLOR),
        ft.Text("📜 实时日志", size=12, weight="bold", color="white"),
        ft.Container(log_list, expand=True)
    ]))

    page.add(
        ft.Row([
            ft.Icon(ft.Icons.SHIELD_MOON, color=CYAN_GLOW, size=28),
            ft.Text("V-GUARD", size=24, weight="900", color="white", italic=True),
            ft.Container(expand=True),
            ft.Text(datetime.now().strftime("%Y-%m-%d %H:%M"), color=TEXT_MUTED, weight="bold")
        ]),
        ft.Container(height=10),
        ft.Row([panel_left, panel_mid, panel_right], expand=True, spacing=20)
    )

    # ==========================================
    # 4. UI 刷新任务 (数据绑定)
    # ==========================================
    async def refresh_task():
        while True:
            total = shared_state.total_risk
            decision = shared_state.decision
            user_bubble_text.value = f"User: {shared_state.asr_text}"

            if decision == "BLOCK":
                state_color = RED_GLOW
                system_title.value = f"● 拦截 (BLOCK) Risk: {total:.2f}"
                system_bubble.bgcolor = "#ff2a2a15"
            elif decision == "WARN":
                state_color = ORANGE_GLOW
                system_title.value = f"● 警告 (WARN) Risk: {total:.2f}"
                system_bubble.bgcolor = "#ffa50015"
            else:
                state_color = GREEN_GLOW
                system_title.value = f"● 放行 (PASS) Risk: {total:.2f}"
                system_bubble.bgcolor = "#34c75915"

            system_icon.color = state_color
            system_title.color = state_color
            system_bubble.border = get_border(2, state_color)

            # 在 refresh_task 内部，确保这部分逻辑是这样写的：
            if shared_state.latest_reports:
                for r in shared_state.latest_reports:
                    if r.module_id == "A":
                        xray_a_val.value = f"Risk: {r.risk_score:.2f}"
                    elif r.module_id == "B":
                        # 关键：确保这里更新的是 xray_b_val
                        xray_b_val.value = f"Risk: {r.risk_score:.2f}"
                        xray_b_val.color = RED_GLOW if r.risk_score > 0.6 else GREEN_GLOW
                    elif r.module_id == "C":
                        xray_c_val.value = f"Risk: {r.risk_score:.2f}"
                ts = datetime.now().strftime("%H:%M:%S")
                log_list.controls.insert(0, ft.Text(f"[{ts}] {decision} | 识别: {shared_state.asr_text[:10]}...",
                                                    color=state_color, size=11, weight="bold"))
                if len(log_list.controls) > 15: log_list.controls.pop()

            page.update()
            await asyncio.sleep(0.5)

    page.run_task(detection_engine_task)
    page.run_task(refresh_task)


if __name__ == "__main__":
    ft.run(main_ui)
'''
# ui/app.py
import flet as ft
import asyncio
from datetime import datetime
from core.state import shared_state
from core.engine import VGuardEngine


BG_COLOR = "#080c11"
PANEL_BG = "#0d131a"
BORDER_COLOR = "#1a2636"
CYAN_GLOW = "#00f2ff"
RED_GLOW = "#ff2a2a"
ORANGE_GLOW = "#ffa500"
GREEN_GLOW = "#34c759"
TEXT_MUTED = "#647b91"


def get_border(width, color):
    side = ft.BorderSide(width, color)
    return ft.Border(top=side, bottom=side, left=side, right=side)


def create_glowing_panel(title, content, glow_color=CYAN_GLOW, expand=True):
    return ft.Container(
        expand=expand, padding=20, bgcolor=PANEL_BG, border_radius=15,
        border=get_border(1, BORDER_COLOR),
        shadow=ft.BoxShadow(spread_radius=0, blur_radius=10, color=f"{glow_color}40", offset=ft.Offset(0, 0)),
        content=ft.Column([
            ft.Text(title, size=14, weight="900", color="white"),
            ft.Divider(height=1, color=BORDER_COLOR),
            content
        ])
    )


async def main_ui(page: ft.Page):
    # ==========================================
    # 0. 挂载真实后台引擎 (已修复引擎调用方法)
    # ==========================================
    async def detection_engine_task():
        engine = VGuardEngine()
        print("🧠 [System] 后台安全决策引擎已激活...")
        while shared_state.is_running:
            try:
                # 你的真实引擎内部已经更新了 shared_state，直接调用 run_pipeline 即可 [cite: 2, 4]
                await engine.run_pipeline()
            except Exception as e:
                print(f"引擎运行错误: {e}")
            await asyncio.sleep(0.8)

    # ==========================================
    # 1. 页面配置
    # ==========================================
    page.title = "V-Guard Pro | 智能座舱防御总控"
    page.bgcolor = BG_COLOR
    page.padding = 20
    page.window_width = 1440
    page.window_height = 850
    page.theme_mode = ft.ThemeMode.DARK

    # ==========================================
    # 2. 交互控制事件
    # ==========================================
    def update_speed(e):
        val = int(e.control.value)
        shared_state.vehicle_speed = val
        speed_text.value = str(val)
        speed_ring.value = min(val / 200.0, 1.0)
        page.update()

    def update_weather(e):
        shared_state.weather = e.control.value
        weather_text.value = f"环境: {e.control.value}"
        page.update()

    def submit_asr(e):
        if asr_input.value:
            shared_state.asr_text = asr_input.value
            page.update()

    def trigger_scenario(speed, weather, text):
        shared_state.vehicle_speed = speed
        shared_state.weather = weather
        shared_state.asr_text = text

        speed_slider.value = speed
        speed_text.value = str(speed)
        speed_ring.value = min(speed / 200.0, 1.0)
        weather_dropdown.value = weather
        weather_text.value = f"环境: {weather}"
        asr_input.value = text
        page.update()

    # ==========================================
    # 3. UI 控件搭建
    # ==========================================

    # --- 左侧：环境与车控模拟 ---
    speed_text = ft.Text(str(shared_state.vehicle_speed), size=60, weight="900", color="white")
    speed_ring = ft.ProgressRing(width=250, height=250, stroke_width=15, value=0.0, color=CYAN_GLOW,
                                 bgcolor=BORDER_COLOR)
    weather_text = ft.Text(f"环境: {shared_state.weather}", size=14, color=GREEN_GLOW, weight="bold")

    speedometer = ft.Stack([
        speed_ring,
        ft.Container(
            content=ft.Column([
                speed_text, ft.Text("km/h", size=16, color=TEXT_MUTED, weight="bold"),
                ft.Text("D", size=24, color=CYAN_GLOW, weight="900"), weather_text
            ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=0),
            alignment=ft.Alignment(0, 0), width=250, height=250
        )
    ])

    speed_slider = ft.Slider(min=0, max=200, divisions=40, value=shared_state.vehicle_speed, active_color=CYAN_GLOW,
                             on_change=update_speed)

    weather_dropdown = ft.Dropdown(
        options=[
            ft.DropdownOption("晴朗"),
            ft.DropdownOption("多云"),
            ft.DropdownOption("暴雨"),
            ft.DropdownOption("暴雪")
        ],
        value=shared_state.weather, width=150, border_color=CYAN_GLOW, color="white",
        on_select=update_weather
    )

    panel_left = create_glowing_panel("CONTEXT SIMULATOR", ft.Column([
        ft.Container(height=10),
        ft.Container(speedometer, alignment=ft.Alignment(0, 0)),
        ft.Container(height=20),
        ft.Text("SPEED CONTROL (车速)", size=12, weight="bold", color="white"),
        speed_slider,
        ft.Row([ft.Text("WEATHER (天气):", size=12, weight="bold", color="white"), weather_dropdown],
               alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
    ]))

    # --- 中侧：核心防御管线 ---
    user_bubble_text = ft.Text("等待语音输入...", size=18, color="white")
    # 修复 padding 弃用警告，直接使用基础数值
    user_bubble = ft.Container(
        content=user_bubble_text, padding=15,
        border_radius=15, border=get_border(2, CYAN_GLOW), bgcolor="#00f2ff10"
    )

    system_icon = ft.Icon(ft.Icons.SHIELD, color=CYAN_GLOW, size=40)
    system_title = ft.Text("● 监控中 (MONITOR)", size=22, weight="900", color=CYAN_GLOW)
    system_reason = ft.Text("系统运行正常", size=16, color="white")

    # 修复 padding 弃用警告，直接使用基础数值
    system_bubble = ft.Container(
        content=ft.Row([system_icon, ft.Column([system_title, system_reason], spacing=2)]),
        padding=20, border_radius=15,
        border=get_border(2, CYAN_GLOW), bgcolor="#00f2ff10"
    )

    def xray_box(title, icon_name):
        val_text = ft.Text("Risk: 0.00", size=10, color=TEXT_MUTED)
        box = ft.Container(
            content=ft.Column([
                ft.Icon(icon_name, color=CYAN_GLOW, size=30),
                ft.Text(title, size=12, weight="bold", color="white"), val_text
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=15, border=get_border(1, CYAN_GLOW), border_radius=10, bgcolor=BORDER_COLOR
        )
        return box, val_text

    box_a, xray_a_val = xray_box("Audio (声学)", ft.Icons.MULTILINE_CHART)
    box_b, xray_b_val = xray_box("ASR (行为)", ft.Icons.SPEED)
    box_c, xray_c_val = xray_box("Semantic (语义)", ft.Icons.RULE)

    xray_row = ft.Container(
        content=ft.Column([
            ft.Text("X-RAY VIEW (透视探测器)", size=12, weight="bold", color="white"),
            ft.Row([box_a, ft.Icon(ft.Icons.ARROW_FORWARD, color=CYAN_GLOW), box_b,
                    ft.Icon(ft.Icons.ARROW_FORWARD, color=CYAN_GLOW), box_c], alignment=ft.MainAxisAlignment.CENTER)
        ]), padding=15, border=get_border(1, BORDER_COLOR), border_radius=10
    )

    panel_mid = create_glowing_panel("V-GUARD DEFENSE PIPELINE", ft.Column([
        ft.Container(height=20), user_bubble, ft.Container(height=40),
        ft.Container(system_bubble, alignment=ft.Alignment(1, 0)), ft.Container(expand=True), xray_row
    ]))

    # --- 右侧：测试注入与预设场景 ---
    asr_input = ft.TextField(label="模拟语音指令注入 (按回车发送)", expand=True, border_color=CYAN_GLOW, color="white",
                             on_submit=submit_asr)
    btn_send = ft.IconButton(icon=ft.Icons.SEND, icon_color=CYAN_GLOW, on_click=submit_asr)

    def scenario_button(title, subtitle, color, speed, weather, text):
        return ft.Container(
            content=ft.Column(
                [ft.Text(title, weight="bold", color=color), ft.Text(subtitle, size=10, color=TEXT_MUTED)], spacing=2),
            padding=15, border=get_border(1, color), border_radius=8, bgcolor=f"{color}10",
            on_click=lambda _: trigger_scenario(speed, weather, text), ink=True
        )

    log_list = ft.ListView(expand=True, spacing=5, auto_scroll=True)

    panel_right = create_glowing_panel("ATTACK & TELEMETRY", ft.Column([
        ft.Text("COMMAND INJECTION (手动测试)", size=12, weight="bold", color="white"),
        ft.Row([asr_input, btn_send]),
        ft.Divider(height=10, color=BORDER_COLOR),
        ft.Text("PRESET SCENARIOS (一键演示)", size=12, weight="bold", color="white"),
        scenario_button("🚗 高速越权危险", "120km/h + 开启后备箱", RED_GLOW, 120, "晴朗", "打开后备箱"),
        ft.Container(height=5),
        scenario_button("🌧️ 极端天气冲突", "40km/h + 暴雨 + 关雨刮", ORANGE_GLOW, 40, "暴雨", "关闭感应雨刮器"),
        ft.Container(height=5),
        scenario_button("🅿️ 停车合规放行", "0km/h + 开启后备箱", GREEN_GLOW, 0, "晴朗", "打开后备箱"),
        ft.Divider(height=10, color=BORDER_COLOR),
        ft.Text("📜 实时日志", size=12, weight="bold", color="white"),
        ft.Container(log_list, expand=True)
    ]))

    page.add(
        ft.Row([
            ft.Icon(ft.Icons.SHIELD_MOON, color=CYAN_GLOW, size=28),
            ft.Text("V-GUARD PRO", size=24, weight="900", color="white", italic=True),
            ft.Container(expand=True),
            ft.Text(datetime.now().strftime("%H:%M"), color=TEXT_MUTED, weight="bold")
        ]),
        ft.Container(height=10),
        ft.Row([panel_left, panel_mid, panel_right], expand=True, spacing=20)
    )

    async def refresh_task():
        while True:
            total = shared_state.total_risk
            decision = shared_state.decision

            user_bubble_text.value = f"User: {shared_state.asr_text}"

            if decision == "BLOCK":
                state_color = RED_GLOW
                system_title.value = f"● 拦截 (BLOCK) Risk: {total:.2f}"
                system_bubble.bgcolor = "#ff2a2a15"
            elif decision == "WARN":
                state_color = ORANGE_GLOW
                system_title.value = f"● 警告 (WARN) Risk: {total:.2f}"
                system_bubble.bgcolor = "#ffa50015"
            else:
                state_color = GREEN_GLOW
                system_title.value = f"● 放行 (PASS) Risk: {total:.2f}"
                system_bubble.bgcolor = "#34c75915"

            system_icon.color = state_color
            system_title.color = state_color
            system_bubble.border = get_border(2, state_color)

            if shared_state.latest_reports:
                for r in shared_state.latest_reports:
                    if r.module_id == "A":
                        xray_a_val.value = f"Risk: {r.risk_score:.2f}"
                    elif r.module_id == "B":
                        xray_b_val.value = f"Risk: {r.risk_score:.2f}"
                    elif r.module_id == "C":
                        xray_c_val.value = f"Risk: {r.risk_score:.2f}"
                        # 确保提取出底层的拦截原因 [cite: 9, 30]
                        system_reason.value = r.reason if decision != "PASS" else "指令合规，正在执行"

                ts = datetime.now().strftime("%H:%M:%S")
                log_list.controls.insert(0, ft.Text(
                    f"[{ts}] {decision} | 综合风险: {total:.2f}",
                    color=state_color, size=11, weight="bold"
                ))
                if len(log_list.controls) > 15: log_list.controls.pop()

            page.update()
            await asyncio.sleep(0.5)

    page.run_task(detection_engine_task)
    page.run_task(refresh_task)


if __name__ == "__main__":
    ft.run(main_ui)
'''
'''
import flet as ft
import asyncio
from datetime import datetime
from core.state import shared_state
from core.engine import VGuardEngine


# 辅助函数：生成透明颜色
def get_alpha_color(color_hex, opacity):
    try:
        alpha = int(opacity * 255)
        return f"#{alpha:02x}{color_hex.lstrip('#')}"
    except:
        return color_hex


async def main_ui(page: ft.Page):
    # --- 1. 后台引擎任务 (生产数据) ---
    async def detection_engine_task():
        engine = VGuardEngine()
        print("[System] 后台安全决策引擎激活")
        while shared_state.is_running:
            reports = engine.generate_mock_reports()
            total_risk, decision = await engine.run_fusion(reports)
            # 同步到状态机
            shared_state.latest_reports = reports
            shared_state.total_risk = total_risk
            shared_state.decision = decision
            await asyncio.sleep(0.8)

    # --- 2. 页面配置 ---
    page.title = "V-Guard Pro | 智能座舱安全防御总控"
    page.bgcolor = "#000000"
    page.theme_mode = ft.ThemeMode.DARK
    page.window_width = 1300
    page.window_height = 900

    ACCENT_CYAN = "#40c4ff"
    ACCENT_RED = "#ff3b30"
    ACCENT_GREEN = "#34c759"
    ACCENT_AMBER = "#ff9500"
    TEXT_SUBTLE = "#8e8e93"

    # --- 3. UI 控件初始化 (确保引用正确) ---
    risk_ring = ft.ProgressRing(width=210, height=210, stroke_width=16, value=0, color=ACCENT_CYAN)
    risk_val = ft.Text("0%", size=54, weight="900")

    # 重点：初始化时确保有字，不要显示空
    asr_display = ft.Text("等待系统唤醒...", size=22, weight="bold")
    speed_display = ft.Text("0 km/h", size=24, weight="bold")
    weather_display = ft.Text("晴朗", size=24, weight="bold")

    def create_card(title, color):
        return ft.Container(
            expand=True, bgcolor="#1c1c1e", padding=15, border_radius=15,
            content=ft.Column([
                ft.Row([ft.Icon(ft.Icons.SHIELD, color=color, size=18), ft.Text(title, size=14, weight="bold")]),
                ft.ProgressBar(value=0, color=color, bgcolor="#000000", height=8),
                ft.Text("系统监控中", size=11, color=TEXT_SUBTLE)
            ])
        )

    card_a = create_card("声学物理层探测", ACCENT_AMBER)
    card_b = create_card("ASR 模型行为分析", ACCENT_CYAN)
    card_c = create_card("语义与状态校验", ACCENT_GREEN)

    log_list = ft.ListView(expand=True, spacing=10, auto_scroll=True)

    # --- 4. 键盘监听 (解决“在哪按”的问题) ---
    def on_keyboard(e: ft.KeyboardEvent):
        # 调试信息：会在你的 PyCharm 终端打印，看到它说明按键成功了
        print(f"收到按键指令: {e.key}")

        if e.key == "H":  # 模拟高速场景
            shared_state.vehicle_speed = 120
            shared_state.asr_text = "确认开启自动驾驶模式"
            shared_state.weather = "晴天"
            shared_state.intent_label = "动力请求"
        elif e.key == "P":  # 模拟停车场景
            shared_state.vehicle_speed = 0
            shared_state.asr_text = "请打开后备箱"
            shared_state.weather = "多云"
            shared_state.intent_label = "设备执行"
        elif e.key == "W":  # 模拟暴雨场景
            shared_state.vehicle_speed = 40
            shared_state.asr_text = "关闭感应雨刮器"
            shared_state.weather = "暴雨"
            shared_state.intent_label = "安全冲突"
        elif e.key=="S":
            shared_state.vehicle_speed=30
            shared_state.asr_text="加速"
            shared_state.weather="暴雪"
            shared_state.intent_label="安全冲突"

        # 立即强制刷新文字控件，不等循环
        asr_display.value = shared_state.asr_text
        speed_display.value = f"{shared_state.vehicle_speed} km/h"
        weather_display.value = shared_state.weather
        page.update()

    page.on_keyboard_event = on_keyboard

    # --- 5. 布局组装 ---
    page.add(
        ft.Row([
            ft.Icon(ft.Icons.SHIELD_MOON, color=ACCENT_CYAN),
            ft.Text("V-GUARD PRO MONITOR", size=20, weight="900"),
            ft.Container(expand=True),
            ft.Text("请点击窗口后按 S/W/P/H 键", size=12, color=ACCENT_CYAN, weight="bold")
        ]),
        ft.Container(height=20),
        ft.Row([
            # 左侧：风险仪表盘
            ft.Container(
                expand=3, bgcolor="#111112", padding=30, border_radius=25,
                content=ft.Column([
                    ft.Stack([risk_ring, ft.Container(risk_val, alignment=ft.Alignment(0, 0), width=210, height=210)]),
                    ft.Container(height=20),
                    ft.Row([card_a, card_b, card_c], spacing=15)
                ], horizontal_alignment="center")
            ),
            # 右侧：实时看板与日志流
            ft.Column([
                ft.Container(
                    content=ft.Column([
                        ft.Text("当前语音识别结果", size=12, color=TEXT_SUBTLE),
                        asr_display,
                        ft.Divider(color="#2c2c2e"),
                        ft.Row([
                            ft.Column([ft.Text("实时车速", size=12, color=TEXT_SUBTLE), speed_display]),
                            ft.Container(width=40),
                            ft.Column([ft.Text("天气环境", size=12, color=TEXT_SUBTLE), weather_display]),
                        ])
                    ]),
                    bgcolor="#111112", padding=20, border_radius=20, width=400
                ),
                ft.Container(
                    expand=True, bgcolor="#111112", padding=20, border_radius=20, width=400,
                    content=ft.Column([
                        ft.Text("📜 安全决策日志流", size=14, weight="bold", color=ACCENT_CYAN),
                        log_list
                    ])
                )
            ], expand=2, spacing=15)
        ], expand=True)
    )

    # --- 6. UI 实时刷新循环 (同步所有数据) ---
    async def refresh_task():
        while True:
            # 同步圆环数值
            total = shared_state.total_risk
            risk_ring.value = total
            risk_ring.color = ACCENT_RED if total > 0.6 else ACCENT_AMBER if total > 0.3 else ACCENT_CYAN
            risk_val.value = f"{int(total * 100)}%"

            # 同步看板文字 (确保万无一失)
            asr_display.value = shared_state.asr_text
            speed_display.value = f"{shared_state.vehicle_speed} km/h"
            weather_display.value = shared_state.weather

            # 处理日志
            if shared_state.latest_reports:
                ts = datetime.now().strftime("%H:%M:%S")
                log_list.controls.insert(0, ft.Text(
                    f"[{ts}] {shared_state.decision} | 风险分: {total:.2f}",
                    color=risk_ring.color, size=12, weight="bold"
                ))
                if len(log_list.controls) > 15: log_list.controls.pop()

                # 更新 A/B/C 卡片
                for r in shared_state.latest_reports:
                    target = card_a if r.module_id == "A" else card_b if r.module_id == "B" else card_c
                    target.content.controls[1].value = r.risk_score
                    # 直接读取 r.reason，因为它是 RiskReport 的直接属性
                    target.content.controls[2].value = f"原因: {r.reason}"
                    # 同时更新进度条颜色
                    target.content.controls[1].value = r.risk_score
                    target.content.controls[1].color = ACCENT_RED if r.risk_score > 0.6 else ACCENT_GREEN

            page.update()
            await asyncio.sleep(0.5)

    # --- 7. 启动所有异步后台任务 ---
    page.run_task(detection_engine_task)  # 启动算法引擎
    page.run_task(refresh_task)  # 启动 UI 刷新
'''
'''
#字段显示不全
import flet as ft
import asyncio
from datetime import datetime
from core.state import shared_state
from core.engine import VGuardEngine


# 辅助函数：生成透明颜色
def get_alpha_color(color_hex, opacity):
    try:
        alpha = int(opacity * 255)
        return f"#{alpha:02x}{color_hex.lstrip('#')}"
    except:
        return color_hex


async def main_ui(page: ft.Page):
    # --- 1. 后台引擎任务 (生产者) ---
    async def detection_engine_task():
        engine = VGuardEngine()
        print("🧠 [System] 后台安全决策引擎已激活...")
        while shared_state.is_running:
            # 获取 A/B/C 的模拟/真实报告
            reports = engine.generate_mock_reports()
            # 融合计算总风险
            total_risk, decision = await engine.run_fusion(reports)

            # 更新全局状态机
            shared_state.latest_reports = reports
            shared_state.total_risk = total_risk
            shared_state.decision = decision
            await asyncio.sleep(0.8)  # 略低于UI刷新频率，保证数据新鲜

    # --- 2. 页面配置与常量 ---
    page.title = "V-Guard Pro | 智能座舱安全防御总控"
    page.bgcolor = "#000000"
    page.window_width = 1300
    page.window_height = 900

    ACCENT_CYAN = "#40c4ff"
    ACCENT_RED = "#ff3b30"
    ACCENT_GREEN = "#34c759"
    ACCENT_AMBER = "#ff9500"
    TEXT_SUBTLE = "#8e8e93"

    # --- 3. UI 组件初始化 ---
    # 风险环尺寸调整为 210 以防止超出
    risk_ring = ft.ProgressRing(width=210, height=210, stroke_width=16, value=0, color=ACCENT_CYAN)
    risk_val = ft.Text("0%", size=54, weight="900")

    # 实时看板数据
    asr_text = ft.Text(shared_state.asr_text, size=22, weight="bold")
    speed_text = ft.Text("0 km/h", size=24, weight="bold")
    weather_text = ft.Text("晴天", size=24, weight="bold")

    def create_card(title, color):
        return ft.Container(
            expand=True, bgcolor="#1c1c1e", padding=15, border_radius=15,
            content=ft.Column([
                ft.Row([ft.Icon(ft.Icons.SHIELD, color=color, size=18), ft.Text(title, size=14, weight="bold")]),
                ft.ProgressBar(value=0, color=color, bgcolor="#000000", height=8),
                ft.Text("监控中...", size=11, color=TEXT_SUBTLE)
            ])
        )

    card_a = create_card("声学探测", ACCENT_AMBER)
    card_b = create_card("ASR行为", ACCENT_CYAN)
    card_c = create_card("语义校验", ACCENT_GREEN)

    # 决策日志列表
    log_list = ft.ListView(expand=True, spacing=10, auto_scroll=True)

    # --- 4. 键盘监听 (场景模拟器) ---
    def on_keyboard(e: ft.KeyboardEvent):
        if e.key == "S":  # 模拟高速
            shared_state.vehicle_speed = 120
            shared_state.asr_text = "确认开启自动驾驶"
            shared_state.intent_label = "动力控制"
        elif e.key == "P":  # 模拟停车
            shared_state.vehicle_speed = 0
            shared_state.asr_text = "打开后备箱"
            shared_state.intent_label = "设备执行"
        elif e.key == "W":  # 模拟暴雨
            shared_state.weather = "暴雨"
            shared_state.asr_text = "关闭所有雨刮器"
        page.update()

    page.on_keyboard_event = on_keyboard

    # --- 5. 布局组装 ---
    page.add(
        ft.Row([
            ft.Icon(ft.Icons.SHIELD_MOON, color=ACCENT_CYAN),
            ft.Text("V-GUARD PRO MONITOR", size=20, weight="900"),
            ft.Container(expand=True),
            ft.Text("上帝模式: 按 S/W/P 键切换场景", size=12, color=TEXT_SUBTLE)
        ]),
        ft.Container(height=20),
        ft.Row([
            # 左：风险评估仪表盘
            ft.Container(
                expand=3, bgcolor="#111112", padding=30, border_radius=25,
                content=ft.Column([
                    ft.Stack([risk_ring, ft.Container(risk_val, alignment=ft.Alignment(0, 0), width=210, height=210)]),
                    ft.Container(height=20),
                    ft.Row([card_a, card_b, card_c], spacing=15)
                ], horizontal_alignment="center")
            ),
            # 右：实时数据看板
            ft.Column([
                ft.Container(
                    content=ft.Column([
                        ft.Text("当前语音识别结果", size=12, color=TEXT_SUBTLE),
                        asr_text,
                        ft.Divider(color="#2c2c2e"),
                        ft.Row([
                            ft.Column([ft.Text("实时车速", size=12, color=TEXT_SUBTLE), speed_text]),
                            ft.Container(width=40),
                            ft.Column([ft.Text("天气环境", size=12, color=TEXT_SUBTLE), weather_text]),
                        ])
                    ]),
                    bgcolor="#111112", padding=20, border_radius=20, width=400
                ),
                # 修复后的日志流区
                ft.Container(
                    expand=True, bgcolor="#111112", padding=20, border_radius=20, width=400,
                    content=ft.Column([
                        ft.Text("📜 安全决策日志流", size=14, weight="bold", color=ACCENT_CYAN),
                        log_list  # 日志被放在 expand 的容器中，确保可见
                    ])
                )
            ], expand=2, spacing=15)
        ], expand=True)
    )

    # --- 6. UI 刷新任务 (消费者) ---
    async def refresh_task():
        while True:
            # 从全局状态同步数据到 UI 控件
            total = shared_state.total_risk
            risk_ring.value = total
            risk_ring.color = ACCENT_RED if total > 0.6 else ACCENT_AMBER if total > 0.3 else ACCENT_CYAN
            risk_val.value = f"{int(total * 100)}%"

            speed_text.value = f"{shared_state.vehicle_speed} km/h"
            weather_text.value = shared_state.weather
            asr_text.value = shared_state.asr_text

            if shared_state.latest_reports:
                # 实时插入决策日志
                ts = datetime.now().strftime("%H:%M:%S")
                log_list.controls.insert(0, ft.Text(
                    f"[{ts}] {shared_state.decision} | Risk: {total:.2f}",
                    color=risk_ring.color, size=12, weight="bold"
                ))
                if len(log_list.controls) > 20: log_list.controls.pop()

                # 更新 A/B/C 子卡片
                for r in shared_state.latest_reports:
                    target = card_a if r.module_id == "A" else card_b if r.module_id == "B" else card_c
                    target.content.controls[1].value = r.risk_score
                    target.content.controls[2].value = f"原因: {r.evidence.get('reason', '正常')}"

            page.update()
            await asyncio.sleep(0.5)

    # --- 7. 启动所有异步后台任务 ---
    page.run_task(detection_engine_task)  # 启动引擎
    page.run_task(refresh_task)  # 启动 UI 刷新
'''
'''
#多功能无显示
import flet as ft
import asyncio
from datetime import datetime
from core.state import shared_state


# 自定义透明色函数
def get_alpha_color(color_hex, opacity):
    try:
        alpha = int(opacity * 255)
        return f"#{alpha:02x}{color_hex.lstrip('#')}"
    except:
        return color_hex


async def main_ui(page: ft.Page):
    # 配置
    page.title = "V-Guard Pro | 智能座舱防御总控"
    page.bgcolor = "#000000"
    page.window_width = 1300
    page.window_height = 900

    # 常量
    ACCENT_CYAN = "#40c4ff"
    ACCENT_RED = "#ff3b30"
    ACCENT_GREEN = "#34c759"
    ACCENT_AMBER = "#ff9500"

    # --- UI 组件定义 ---
    # 1. 核心看板数据
    asr_text = ft.Text(shared_state.asr_text, size=22, weight="bold")
    speed_text = ft.Text("0 km/h", size=24, weight="bold")
    weather_text = ft.Text("晴天", size=24, weight="bold")

    # 2. 风险环
    risk_ring = ft.ProgressRing(width=210, height=210, stroke_width=16, value=0, color=ACCENT_CYAN)
    risk_val = ft.Text("0%", size=54, weight="900")

    # 3. 模块卡片
    def create_card(title, color):
        return ft.Container(
            expand=True, bgcolor="#1c1c1e", padding=15, border_radius=15,
            content=ft.Column([
                ft.Row([ft.Icon(ft.Icons.SHIELD, color=color, size=18), ft.Text(title, size=14, weight="bold")]),
                ft.ProgressBar(value=0, color=color, bgcolor="#000000", height=8),
                ft.Text("正常", size=11, color="#8e8e93")
            ])
        )

    card_a, card_b, card_c = create_card("声学探测", ACCENT_AMBER), create_card("ASR行为", ACCENT_CYAN), create_card(
        "语义校验", ACCENT_GREEN)

    # 4. 【修复重点】日志区域
    log_list = ft.ListView(expand=True, spacing=10, auto_scroll=True)

    # --- 布局组装 ---
    page.add(
        # Header
        ft.Row([
            ft.Icon(ft.Icons.SHIELD_MOON, color=ACCENT_CYAN, size=28),
            ft.Text("V-GUARD PRO MONITOR", size=20, weight="900"),
            ft.Container(expand=True),
            ft.Text("按 S/W/P 切换场景", size=12, color="#8e8e93")
        ]),
        ft.Container(height=20),
        # Main Area
        ft.Row([
            # 左：风险评估
            ft.Container(
                expand=3, bgcolor="#111112", padding=30, border_radius=25,
                content=ft.Column([
                    ft.Stack([risk_ring, ft.Container(risk_val, alignment=ft.Alignment(0, 0), width=210, height=210)]),
                    ft.Container(height=20),
                    ft.Row([card_a, card_b, card_c], spacing=15)
                ], horizontal_alignment="center")
            ),
            # 右：实时数据看板
            ft.Column([
                ft.Container(
                    content=ft.Column([
                        ft.Text("当前语音识别结果", size=12, color="#8e8e93"),
                        asr_text,
                        ft.Divider(color="#2c2c2e", height=20),
                        ft.Row([
                            ft.Column([ft.Text("实时车速", size=12, color="#8e8e93"), speed_text]),
                            ft.VerticalDivider(width=20),
                            ft.Column([ft.Text("天气环境", size=12, color="#8e8e93"), weather_text]),
                        ])
                    ]),
                    bgcolor="#111112", padding=20, border_radius=20, width=400
                ),
                # 决策日志容器 (修复显示)
                ft.Container(
                    content=ft.Column([
                        ft.Text("📜 安全决策日志流", size=14, weight="bold", color=ACCENT_CYAN),
                        ft.Container(log_list, expand=True)  # 嵌套 ListView
                    ]),
                    bgcolor="#111112", padding=20, border_radius=20, expand=True, width=400
                )
            ], expand=2, spacing=15)
        ], expand=True)
    )

    # --- 刷新循环 ---
    while True:
        # 更新风险
        total = shared_state.total_risk
        risk_ring.value = total
        risk_ring.color = ACCENT_RED if total > 0.6 else ACCENT_AMBER if total > 0.3 else ACCENT_CYAN
        risk_val.value = f"{int(total * 100)}%"

        # 更新环境看板
        speed_text.value = f"{shared_state.vehicle_speed} km/h"
        weather_text.value = shared_state.weather
        asr_text.value = shared_state.asr_text

        # 更新日志流
        if shared_state.latest_reports:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_list.controls.insert(0, ft.Text(
                f"[{timestamp}] 决策: {shared_state.decision} | 风险分: {total:.2f}",
                size=12, color=risk_ring.color, weight="bold"
            ))
            if len(log_list.controls) > 20: log_list.controls.pop()

            # 更新子模块卡片
            for r in shared_state.latest_reports:
                card = card_a if r.module_id == "A" else card_b if r.module_id == "B" else card_c
                card.content.controls[1].value = r.risk_score
                card.content.controls[2].value = f"原因: {r.evidence.get('reason', '正常')}"

        page.update()
        await asyncio.sleep(1)
'''
'''
import flet as ft
import asyncio
from datetime import datetime
from core.state import shared_state


async def main_ui(page: ft.Page):
    from main import detection_engine_task
    page.run_task(detection_engine_task)

    page.title = "V-Guard Pro | 智能座舱全态势感知"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#000000"
    page.window_width = 1400  # 宽度加宽，更舒展
    page.window_height = 900
    page.padding = 30

    # 颜色与辅助函数
    def get_alpha(c, o):
        return f"#{int(o * 255):02x}{c.lstrip('#')}"

    ACCENT_CYAN = "#40c4ff"
    ACCENT_AMBER = "#ff9500"
    ACCENT_RED = "#ff3b30"
    ACCENT_GREEN = "#34c759"

    # --- UI 组件定义 ---

    # 1. 实时语音与意图看板
    asr_display = ft.Text(shared_state.asr_text, size=24, weight="bold", color="white")
    intent_display = ft.Text(shared_state.intent_label, size=18, color=ACCENT_CYAN, weight="bold")

    # 2. 车辆环境看板组件
    def create_stat_box(label, value, icon):
        return ft.Container(
            content=ft.Column([
                ft.Row([ft.Icon(icon, size=16, color="#8e8e93"), ft.Text(label, size=12, color="#8e8e93")]),
                ft.Text(value, size=24, weight="bold", color="white")
            ], spacing=5),
            expand=True, bgcolor="#1c1c1e", padding=15, border_radius=15
        )

    speed_box = create_stat_box("当前车速", "0 km/h", ft.Icons.SPEED)
    weather_box = create_stat_box("天气环境", "晴天", ft.Icons.CLOUD)
    exec_box = create_stat_box("执行结果", "待机", ft.Icons.SETTINGS_INPUT_COMPONENT)

    # 3. 核心风险圆环
    risk_ring = ft.ProgressRing(width=220, height=220, stroke_width=18, value=0, color=ACCENT_CYAN)
    risk_text = ft.Text("0%", size=54, weight="900")

    # 4. 模块卡片 (A/B/C)
    def create_module_card(title, color):
        return ft.Container(
            expand=True, bgcolor="#1c1c1e", padding=20, border_radius=20,
            content=ft.Column([
                ft.Row([ft.Icon(ft.Icons.SHIELD, color=color, size=20), ft.Text(title, weight="bold")]),
                ft.ProgressBar(value=0, color=color, bgcolor="#000000", height=8),
                ft.Text("原因: 正常", size=11, color="#8e8e93")
            ])
        )

    card_a = create_module_card("物理声学探测", ACCENT_AMBER)
    card_b = create_module_card("ASR 行为分析", ACCENT_CYAN)
    card_c = create_module_card("语义意图校验", ACCENT_GREEN)

    # --- 布局组装 ---
    page.add(
        # Top Header
        ft.Row([
            ft.Icon(ft.Icons.SHIELD_MOON, color=ACCENT_CYAN, size=30),
            ft.Text("V-GUARD PRO", size=24, weight="900"),
            ft.Container(expand=True),
            ft.Text(datetime.now().strftime("%H:%M"), size=16, color="#8e8e93")
        ]),
        ft.Divider(height=40, color="transparent"),

        # Middle: Dashboard & Status
        ft.Row([
            # 左侧：风险仪表盘
            ft.Container(
                expand=4, bgcolor="#111112", padding=40, border_radius=30,
                content=ft.Column([
                    ft.Text("实时安全态势指数", color="#8e8e93"),
                    ft.Stack([
                        risk_ring,
                        ft.Container(risk_text, alignment=ft.Alignment(0, 0), width=220, height=220)
                    ]),
                    ft.Container(height=20),
                    ft.Row([card_a, card_b, card_c], spacing=20)
                ], horizontal_alignment="center")
            ),
            # 右侧：实时数据流
            ft.Column([
                # 语音识别区
                ft.Container(
                    content=ft.Column([
                        ft.Text("实时语音识别 (ASR)", size=12, color="#8e8e93"),
                        asr_display,
                        ft.Divider(color="#2c2c2e"),
                        ft.Text("提取语义意图 (Intent)", size=12, color="#8e8e93"),
                        intent_display,
                    ]),
                    bgcolor="#111112", padding=25, border_radius=25, width=400
                ),
                # 环境状态区
                ft.Row([speed_box, weather_box], width=400),
                exec_box
            ], expand=2, spacing=20)
        ], expand=True),

        # Bottom: 日志流
        ft.Container(height=20),
        ft.Container(
            content=ft.ListView(expand=True, spacing=5),
            expand=1, bgcolor="#111112", padding=20, border_radius=20
        )
    )

    # --- 刷新循环 ---
    while True:
        # 更新风险
        total = shared_state.total_risk
        risk_ring.value = total
        risk_ring.color = ACCENT_RED if total > 0.6 else ACCENT_AMBER if total > 0.3 else ACCENT_CYAN
        risk_text.value = f"{int(total * 100)}%"

        # 更新环境数据
        asr_display.value = shared_state.asr_text
        intent_display.value = f"➡ {shared_state.intent_label}"
        speed_box.content.controls[1].value = f"{shared_state.vehicle_speed} km/h"
        weather_box.content.controls[1].value = shared_state.weather
        exec_box.content.controls[1].value = shared_state.execution_result
        exec_box.content.controls[1].color = ACCENT_RED if "拦截" in shared_state.execution_result else "white"

        # 更新子卡片证据
        if shared_state.latest_reports:
            for r in shared_state.latest_reports:
                card = card_a if r.module_id == "A" else card_b if r.module_id == "B" else card_c
                card.content.controls[1].value = r.risk_score
                card.content.controls[2].value = f"原因: {r.evidence.get('reason', '正常')}"

        page.update()
        await asyncio.sleep(0.5)
'''
'''
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
'''