# datasets/audio_dataset.py
import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    """音频数据集，返回原始波形或特征向量"""
    def __init__(self, root_dir, target_sr=16000, transform=None, ext='.wav'):
        """
        root_dir: 包含音频文件的目录（递归搜索）
        target_sr: 目标采样率
        transform: 可选的预处理函数（如特征提取）
        """
        self.file_paths = []
        for dirpath, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.endswith(ext):
                    self.file_paths.append(os.path.join(dirpath, f))
        self.target_sr = target_sr
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        waveform, sr = torchaudio.load(path)
        # 重采样
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze(0)  # (samples,)

        # 确保音频长度足够（例如至少 1 秒）
        min_samples = self.target_sr
        if waveform.shape[0] < min_samples:
            pad = min_samples - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        # 归一化到 [-1, 1]
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)

        # 应用变换（例如提取特征）
        if self.transform:
            waveform = self.transform(waveform.numpy())  # 转为 numpy 供特征提取器使用
            waveform = torch.from_numpy(waveform).float()

        return waveform