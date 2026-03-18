import sys
import os
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from modules.module1_acoustic.aasist_model import Model as AASIST


def load_model(model_path):
    d_args = {
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.5, 0.5, 0.5],
        "temperatures": [2, 2, 100],
        "first_conv": 128,
    }
    model = AASIST(d_args)
    state_dict = torch.load(model_path, map_location='cpu')
    if 'model' in state_dict:
        state_dict = state_dict['model']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model


def preprocess(audio, expected_length=64600):
    audio = np.array(audio, dtype=np.float32)
    audio = audio.squeeze()  # 确保一维
    if audio.ndim != 1:
        raise ValueError(f"音频应该是1维，但实际是 {audio.ndim} 维")
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / 32768.0
    if len(audio) < expected_length:
        pad = expected_length - len(audio)
        audio = np.pad(audio, (0, pad), mode='constant')
    else:
        start = (len(audio) - expected_length) // 2
        audio = audio[start:start + expected_length]
    tensor = torch.from_numpy(audio).float().unsqueeze(0)  # 只加 batch 维
    return tensor


def test_file(model, wav_path, device='cpu'):
    audio, sr = sf.read(wav_path)
    print(f"\n文件: {os.path.basename(wav_path)}")
    print(f"  采样率: {sr} Hz")
    print(f"  原始形状: {audio.shape}, 数据类型: {audio.dtype}")
    if sr != 16000:
        print("  ⚠️ 采样率不是 16000，结果可能不准")
    # 转为单声道（如果立体声）
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
        print("  已转为单声道")
    # 预处理
    tensor = preprocess(audio).to(device)
    print(f"  预处理后张量形状: {tensor.shape}")
    with torch.no_grad():
        _, output = model(tensor)
        probs = F.softmax(output, dim=-1)
    print(f"  原始 logits: {output.cpu().numpy().flatten()}")
    print(f"  softmax 概率: {probs.cpu().numpy().flatten()}")
    print(f"  类别0 (伪造) 概率: {probs[0, 0].item():.4f}")
    print(f"  类别1 (真人) 概率: {probs[0, 1].item():.4f}")


if __name__ == '__main__':
    # 请修改为你的模型路径
    model_path = r'D:\essay of crypto\AutoSafe-V-Guard\modules\module1_acoustic\models\aasist.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("加载模型中...")
    model = load_model(model_path)
    model.to(device)
    print("模型加载完成。")

    # 请修改为你的三个测试文件路径
    test_file(model, r'D:\essay of crypto\AutoSafe-V-Guard\human.wav', device)  # 真人
    test_file(model, r'D:\essay of crypto\AutoSafe-V-Guard\doubao.wav', device)  # AI 合成
    test_file(model, r'D:\essay of crypto\AutoSafe-V-Guard\resent.wav', device)  # 重放