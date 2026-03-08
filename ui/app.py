import flet as ft
import asyncio
import os
import wave
import pyaudio
import time
from datetime import datetime
import numpy as np  # 新增：用于计算音量
import collections  # 新增：用于循环缓冲区

# 核心架构导入
from core.state import shared_state
from core.engine import VGuardEngine
from modules.module2_ASR.asr_risk_model import ASRRiskModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 样式常量 (保持不变) ---
BG_COLOR = "#080c11"
PANEL_BG = "#0d131a"
BORDER_COLOR = "#1a2636"
CYAN_GLOW = "#00f2ff"
RED_GLOW = "#ff2a2a"
ORANGE_GLOW = "#ffa500"
GREEN_GLOW = "#34c759"
TEXT_MUTED = "#647b91"

# --- 全局音频缓冲区 (16000采样率, 每秒约15.6个CHUNK, 存5秒共约78个CHUNK) ---
# 这个缓冲区是“实时录音”和“声波图”共享的命脉
audio_buffer = collections.deque(maxlen=80)


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


# --- 修改后的录音逻辑：直接从内存提取数据，不再占用麦克风硬件 ---
def save_buffer_to_wav(filename="realtime_mic.wav"):
    CHUNK = 1024
    CHANNELS = 1
    RATE = 16000
    p = pyaudio.PyAudio()

    # 将当前的缓冲区转为列表，保证保存时数据稳定
    frames = list(audio_buffer)

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    p.terminate()
    return filename


async def main_ui(page: ft.Page):
    engine = VGuardEngine()

    # 初始化状态
    if not hasattr(shared_state, 'total_risk'): shared_state.total_risk = 0.0
    if not hasattr(shared_state, 'decision'): shared_state.decision = "PASS"
    if not hasattr(shared_state, 'latest_reports'): shared_state.latest_reports = []
    if not hasattr(shared_state, 'vehicle_speed'): shared_state.vehicle_speed = 0
    if not hasattr(shared_state, 'asr_text'): shared_state.asr_text = "等待输入..."
    if not hasattr(shared_state, 'realtime_volume'): shared_state.realtime_volume = 0.0  # 实时音量

    if not hasattr(shared_state, 'asr_model'):
        try:
            shared_state.asr_model = ASRRiskModel()
        except:
            pass

    page.title = "V-Guard Pro | 智能座舱防御总控"
    page.bgcolor = BG_COLOR
    page.padding = 20
    page.window_width = 1440
    page.window_height = 900
    page.theme_mode = ft.ThemeMode.DARK

    # ==========================================
    # 🌊 成员 D 定制：真实的声波波动图组件
    # ==========================================
    bars = []
    for _ in range(30):
        bars.append(ft.Container(width=4, height=10, bgcolor=CYAN_GLOW, border_radius=5,
                                 animate_size=ft.Animation(0, "decelerate"), opacity=0.4))

    visualizer = ft.Row(controls=bars, alignment="center", vertical_alignment="center", spacing=3, height=60)

    # ==========================================
    # 🌟 三个风险圆圈组件 (保持原逻辑)
    # ==========================================
    def create_risk_circle(title):
        ring = ft.ProgressRing(width=70, height=70, stroke_width=7, value=0.0, color=GREEN_GLOW, bgcolor=BORDER_COLOR)
        val_text = ft.Text("0.00", size=16, weight="bold", color="white")
        reason_text = ft.Text("待机中", size=10, color=TEXT_MUTED, text_align="center")
        return ft.Container(content=ft.Column([
            ft.Text(title, size=12, weight="bold", color="white"),
            ft.Stack([ring, ft.Container(val_text, alignment=ft.Alignment(0, 0), width=70, height=70)]),
            reason_text
        ], horizontal_alignment="center", spacing=5), padding=10, border=get_border(1, BORDER_COLOR), border_radius=10,
            width=130
        ), ring, val_text, reason_text

    box_a, ring_a, val_a, reason_a = create_risk_circle("A: 声学物理层")
    box_b, ring_b, val_b, reason_b = create_risk_circle("B: ASR行为层")
    box_c, ring_c, val_c, reason_c = create_risk_circle("C: 语义状态层")

    visual_evidence_row = ft.Row([box_a, ft.Icon(ft.Icons.ARROW_FORWARD, color=CYAN_GLOW),
                                  box_b, ft.Icon(ft.Icons.ARROW_FORWARD, color=CYAN_GLOW),
                                  box_c], alignment="center")

    async def start_voice_defense(e):
        btn_voice.disabled = True
        btn_voice.text = "正在监听 (请开始说话)..."
        btn_voice.icon_color = RED_GLOW
        page.update()

        # 模拟收集音频缓冲区的时间
        await asyncio.sleep(5)

        btn_voice.text = "安全网关分析中..."
        btn_voice.icon_color = ORANGE_GLOW
        page.update()

        try:
            # 1. 仅做外围处理：将内存流固化为物理文件交接给网关
            audio_file = await asyncio.to_thread(save_buffer_to_wav, "realtime_mic.wav")

            # ========================================================
            # 🚀 工业级防腐层调用：所有算法推断必须下沉到 Engine 中
            # UI 层只负责“发送原始文件”并“接收分析报告”
            # ========================================================

            # (注意: 这里假设你的 engine.analyze_risk 已经重构为接受音频路径)
            # engine 会内部调用 A、B、C 三个模块，B 模块会顺便把识别出的文本放在 metadata 里
            result = await engine.analyze_risk_pipeline(
                audio_path=audio_file,
                speed=shared_state.vehicle_speed
            )



            # 3. 更新全局决策状态
            shared_state.asr_text = result["asr_text"]
            shared_state.total_risk = result["total_risk"]
            shared_state.decision = result["decision"]
            shared_state.latest_reports = result["reports"]

            # 4. 更新 UI 日志渲染
            ts = datetime.now().strftime("%H:%M:%S")
            log_list.controls.insert(0, ft.Text(
                f"[{ts}] {result['decision']} | 风险: {result['total_risk']:.2f} | 耗时: {result.get('latency_ms', 0)}ms",
                color=RED_GLOW if result['decision'] == "BLOCK" else (
                    ORANGE_GLOW if result['decision'] == "REVIEW" else GREEN_GLOW),
                size=12, weight="bold"
            ))
            page.update()

        except Exception as ex:
            print(f"❌ 系统级融合管线异常: {ex}")
            shared_state.asr_text = "网关熔断保护"

        btn_voice.disabled = False
        btn_voice.text = "开启语音防御监测"
        btn_voice.icon_color = CYAN_GLOW
        page.update()

    # --- UI 布局代码 (基本保持原样，仅嵌入声波图) ---
    def update_speed(e):
        val = int(e.control.value)
        shared_state.vehicle_speed = val
        speed_text.value = str(val)
        speed_ring.value = min(val / 200.0, 1.0)
        page.update()

    speed_text = ft.Text(str(shared_state.vehicle_speed), size=60, weight="900", color="white")
    speed_ring = ft.ProgressRing(width=250, height=250, stroke_width=15, value=0.0, color=CYAN_GLOW,
                                 bgcolor=BORDER_COLOR)
    speedometer = ft.Stack(
        [speed_ring, ft.Container(content=ft.Column([speed_text, ft.Text("km/h", size=16, color=TEXT_MUTED)],
                                                    alignment="center", horizontal_alignment="center"), width=250,
                                  height=250)])
    speed_slider = ft.Slider(min=0, max=200, divisions=40, value=shared_state.vehicle_speed, on_change=update_speed)

    panel_left = create_glowing_panel("CONTEXT SIMULATOR", ft.Column(
        [ft.Container(height=10), ft.Container(speedometer, alignment=ft.Alignment(0, 0)),
         ft.Container(height=20), speed_slider]))

    user_bubble_text = ft.Text("等待输入...", size=18, color="white")
    user_bubble = ft.Container(content=user_bubble_text, padding=15, border_radius=15, border=get_border(2, CYAN_GLOW),
                               bgcolor="#00f2ff10")

    system_title = ft.Text("● 监控中 (MONITOR)", size=22, weight="900", color=CYAN_GLOW)
    system_reason = ft.Text("系统正常运行中", size=14, color=TEXT_MUTED)
    system_bubble = ft.Container(content=ft.Row([ft.Icon(ft.Icons.SHIELD, color=CYAN_GLOW, size=40),
                                                 ft.Column([system_title, system_reason], spacing=2)]),
                                 padding=20, border_radius=15, border=get_border(2, CYAN_GLOW), bgcolor="#00f2ff10")

    panel_mid = create_glowing_panel("V-GUARD DEFENSE PIPELINE", ft.Column([
        ft.Container(height=10),
        user_bubble,
        ft.Container(height=5),
        ft.Text("LIVE AUDIO SPECTRUM", size=10, color=TEXT_MUTED, weight="bold"),
        visualizer,  # 插入声波图
        ft.Container(height=5),
        system_bubble,
        ft.Container(height=20),
        visual_evidence_row,
        ft.Container(expand=True)
    ]))

    btn_voice = ft.ElevatedButton("开启语音防御监测", icon=ft.Icons.MIC, color=CYAN_GLOW, on_click=start_voice_defense,
                                  height=50)
    log_list = ft.ListView(expand=True, spacing=5, auto_scroll=True)
    panel_right = create_glowing_panel("ATTACK & TELEMETRY",
                                       ft.Column([btn_voice, ft.Divider(height=20, color=BORDER_COLOR),
                                                  ft.Text("实时防御日志", size=12, weight="bold"),
                                                  ft.Container(log_list, expand=True)]))

    page.add(ft.Row([ft.Icon(ft.Icons.SHIELD_MOON, color=CYAN_GLOW), ft.Text("V-GUARD PRO", size=24, weight="900")],
                    alignment="center"),
             ft.Container(height=10), ft.Row([panel_left, panel_mid, panel_right], expand=True, spacing=20))

    # ==========================================
    # 🧠 后台任务 1: 真实的音频流监听 (核心)
    # ==========================================
    async def audio_monitor_task():
        CHUNK, RATE = 1024, 16000
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,input_device_index=4, frames_per_buffer=CHUNK)
        last_v=0.0
        try:
            while True:
                data = await asyncio.to_thread(stream.read, CHUNK, exception_on_overflow=False)
                # 存入缓冲区
                audio_buffer.append(data)
                # 计算实时音量
                samples = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(np.mean(samples.astype(float) ** 2))

                # --- 核心改进：噪声门限 (Noise Gate) ---
                # 如果 RMS 小于 3000，认为那是环境噪音，直接归零
                if rms < 3000:
                    target_v = 0.0
                else:
                    # 动态映射公式：让 3000-25000 映射到 0.0-1.0
                    target_v = min((rms - 3000) / 22000.0, 1.0)

                # 使用极小的平滑系数，增加实时感 (0.6 新值 + 0.4 旧值)
                shared_state.realtime_volume = target_v * 0.6 + (shared_state.realtime_volume * 0.4)
                await asyncio.sleep(0.01)
        except Exception as e:
            print(f"Mic Error: {e}")
        finally:
            stream.stop_stream();
            stream.close();
            p.terminate()

    # ==========================================
    # 🧠 后台任务 2: UI 刷新任务 (含声波跳动)
    # ==========================================
    async def refresh_task():
        while True:
            # 更新声波图
            vol = shared_state.realtime_volume
            for i, bar in enumerate(bars):
                dist = abs(i - 15) / 15.0
                sensitivity = 1.0 - dist
                target_h = 5 + (vol * 120 * sensitivity)
                bar.height = target_h
                bar.bgcolor = RED_GLOW if shared_state.decision == "BLOCK" else CYAN_GLOW
                bar.opacity = 0.3 + (vol * 0.7)

            # 更新其他风险 UI
            total, decision = shared_state.total_risk, shared_state.decision
            user_bubble_text.value = f"User: {shared_state.asr_text}"
            system_title.value = f"● {decision} Risk: {total:.2f}"
            system_title.color = RED_GLOW if decision == "BLOCK" else (
                ORANGE_GLOW if decision == "REVIEW" else GREEN_GLOW)

            if shared_state.latest_reports:
                for r in shared_state.latest_reports:
                    color = GREEN_GLOW if r.risk_score < 0.3 else (ORANGE_GLOW if r.risk_score < 0.6 else RED_GLOW)
                    if r.module_id == "A":
                        ring_a.value, val_a.value, ring_a.color, val_a.color, reason_a.value = r.risk_score, f"{r.risk_score:.2f}", color, color, r.reason[
                                                                                                                                                  :15]
                    elif r.module_id == "B":
                        ring_b.value, val_b.value, ring_b.color, val_b.color, reason_b.value = r.risk_score, f"{r.risk_score:.2f}", color, color, r.reason[
                                                                                                                                                  :15]
                    elif r.module_id == "C":
                        ring_c.value, val_c.value, ring_c.color, val_c.color, reason_c.value = r.risk_score, f"{r.risk_score:.2f}", color, color, r.reason[
                                                                                                                                                  :15]

            page.update()
            await asyncio.sleep(0.05)  # 20FPS 丝滑跳动

    page.run_task(audio_monitor_task)
    page.run_task(refresh_task)


if __name__ == "__main__":
    ft.app(target=main_ui)
