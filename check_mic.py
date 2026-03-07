import pyaudio
p = pyaudio.PyAudio()
print("\n--- 可用的音频输入设备列表 ---")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f"ID {i}: {info['name']}")
p.terminate()
print("----------------------------\n")