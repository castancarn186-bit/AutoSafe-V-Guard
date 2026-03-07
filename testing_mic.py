import pyaudio
import numpy as np
import time

p = pyaudio.PyAudio()
# 探测所有输入设备
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f"\n正在测试设备 ID {i}: {info['name']}")
        try:
            # 尝试用最常用的 44100Hz 打开
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100,
                            input=True, input_device_index=i, frames_per_buffer=1024)

            print("--- 请对着麦克风拍打或大喊 2 秒 ---")
            max_rms = 0
            for _ in range(20):  # 测试 2 秒
                data = stream.read(1024, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(np.mean(samples.astype(float) ** 2))
                if rms > max_rms: max_rms = rms

            stream.stop_stream()
            stream.close()
            print(f">>> 该设备检测到的最大音量值: {max_rms:.2f}")
            if max_rms > 100:
                print("✅ 这是一个活跃设备！")
            else:
                print("❌ 该设备几乎没有声音输入。")
        except Exception as e:
            print(f"⚠️ 无法打开该设备: {e}")

p.terminate()