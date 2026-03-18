import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np

fs = 16000       # 采样率 16kHz
duration = 5     # 秒
print("开始录制...")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()        # 等待录制完成
print("录制结束")

# 保存为 WAV 文件
wav.write('resent.wav', fs, recording)