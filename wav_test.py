import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav


def record_sample(filename, duration=5, sr=16000):
    print(f"🎤 录制开始 (时长 {duration}秒)... 请说话")
    # 录制单声道音频 [cite: 34, 36]
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # 等待录制结束

    # 归一化并保存 [cite: 30]
    wav.write(filename, sr, (recording * 32767).astype(np.int16))
    print(f"💾 已保存至: {filename}")


if __name__ == "__main__":
    record_sample("test_doubao_01.wav")