import os
import wave
import pyaudio

def record_audio(filename="test.wav", duration=15
                 ):
    """
    录制一段音频并保存为 test.wav
    参数:
        filename: 保存的文件名

        duration: 录音时长（秒）
    """
    # 设定参数：16kHz, 单声道, 16bit (Whisper 标准格式) [cite: 187, 211]
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024

    audio = pyaudio.PyAudio()

    print(f"--- 准备录音 ---")
    print(f"请在按下 Enter 后开始说话，录音将持续 {duration} 秒...")
    input("按回车键开始...")

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print(">>> 正在录音... 请说话 (例如：'请打开后备箱')")
    frames = []

    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("--- 录音结束 ---")

    # 停止录音
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # 保存为 wav 文件
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"文件已保存至: {os.path.abspath(filename)}")

if __name__ == "__main__":
    record_audio()