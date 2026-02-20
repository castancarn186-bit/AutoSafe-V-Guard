import os
import wave
import pyaudio
import numpy as np


class VoiceCapture:
    """工程级音频采集器：对接麦克风并输出标准格式文件"""

    def __init__(self, rate=16000, chunk=1024):
        self.rate = rate
        self.chunk = chunk
        self.format = pyaudio.paInt16
        self.channels = 1
        self.p = pyaudio.PyAudio()

    def record(self, seconds=3, save_path="temp_voice.wav"):
        stream = self.p.open(format=self.format, channels=self.channels,
                             rate=self.rate, input=True,
                             frames_per_buffer=self.chunk)
        frames = []
        for _ in range(0, int(self.rate / self.chunk * seconds)):
            frames.append(stream.read(self.chunk))

        stream.stop_stream()
        stream.close()

        with wave.open(save_path, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
        return save_path