# core/sim_env.py
import asyncio
from core.state import shared_state

class VehicleSimulator:
    async def run(self):
        print("[Sim] 环境模拟器已启动(按逻辑自动循环)")
        while shared_state.is_running:
            # 模拟逻辑：如果正在拦截，执行结果就显示“已阻断”
            if "BLOCK" in shared_state.decision:
                shared_state.execution_result = "指令被拦截"
            elif "PASS" in shared_state.decision and shared_state.asr_text != "等待语音输入...":
                shared_state.execution_result = "执行成功"
            else:
                shared_state.execution_result = "待机"

            await asyncio.sleep(0.5)

    def manual_trigger(self, key_event):
        """
        你可以后续在这里绑定键盘事件
        比如：按下 'W' 键切换天气为 '雨天'
        """
        if key_event == "R":
            shared_state.weather = "暴雨"
            print("场景切换：暴雨模式")
        elif key_event == "W":
            shared_state.weather = "雾霾"
            print("场景切换：雾霾模式")
        elif key_event == "S":
            shared_state.weather = "暴雪"
            print("场景切换：暴雪模式")