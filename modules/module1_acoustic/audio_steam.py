"""
实时音频采集模块
- 16kHz单声道
- 环形缓冲区 (Ring Buffer)
- 分块输出（20ms窗口）
"""
import numpy as np
import sounddevice as sd
from collections import deque
import threading
import time

class AudioStream:
    """
    音频流采集器，使用回调函数填充环形缓冲区。
    提供 get_chunk() 方法获取最新的音频块。
    """
    def __init__(self, samplerate=16000, blocksize=320, channels=1, buffer_seconds=2):
        """
        :param samplerate: 采样率 (Hz)
        :param blocksize:  每次回调的帧数 (对应20ms: 16000*0.02 = 320)
        :param channels:   通道数 (1)
        :param buffer_seconds: 环形缓冲区时长 (秒)
        """
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = channels
        self.buffer_size = int(samplerate * buffer_seconds)  # 总样本数
        self.buffer = deque(maxlen=self.buffer_size)        # 环形缓冲区
        self.lock = threading.Lock()
        self.stream = None
        self.running = False

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice 回调函数，将数据推入缓冲区"""
        if status:
            print(f"Audio callback status: {status}")
        # indata shape: (frames, channels)
        with self.lock:
            # 将新数据添加到 deque 右侧
            self.buffer.extend(indata[:, 0])  # 取单声道

    def start(self):
        """启动音频流"""
        if self.running:
            return
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            channels=self.channels,
            callback=self._audio_callback,
            dtype='float32'
        )
        self.stream.start()
        self.running = True

    def stop(self):
        """停止音频流"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.running = False

    def get_chunk(self, chunk_samples=None):
        """
        获取最新的一段音频块（默认 chunk_samples = blocksize）
        如果缓冲区数据不足，返回 None 或填充零？
        实际使用中，系统应保证缓冲区有足够数据，若不足则等待。
        """
        if chunk_samples is None:
            chunk_samples = self.blocksize
        with self.lock:
            if len(self.buffer) < chunk_samples:
                # 数据不足，返回 None
                return None
            # 取出最后 chunk_samples 个样本（即最新数据）
            # deque 不支持直接切片，转换为 list 或使用 collections.deque 的右侧 pop
            # 但 pop 会移除数据，我们希望保留缓冲区，所以复制一份
            buf_list = list(self.buffer)
            chunk = np.array(buf_list[-chunk_samples:], dtype=np.float32)
        return chunk

    def get_all(self):
        """获取整个缓冲区数据（用于保存或调试）"""
        with self.lock:
            return np.array(self.buffer, dtype=np.float32)