"""
异常检测模型训练脚本
- 从指定目录加载正常音频和异常音频
- 提取特征
- 训练 One-Class SVM (或 Isolation Forest)
- 保存模型
"""
import os
import numpy as np
import librosa
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import joblib
from feature_extractor import FeatureExtractor

def load_audio_files(directory, sr=16000, ext='.wav'):
    """加载目录下所有音频文件，返回音频列表"""
    audio_list = []
    for file in os.listdir(directory):
        if file.endswith(ext):
            path = os.path.join(directory, file)
            audio, _ = librosa.load(path, sr=sr)
            audio_list.append(audio)
    return audio_list

def extract_features_from_audios(audio_list, feature_extractor):
    """从音频列表批量提取特征"""
    features = []
    for audio in audio_list:
        feat = feature_extractor.extract(audio)
        features.append(feat)
    return np.array(features)

def main():
    # 配置
    normal_dir = "data/normal_audio"
    abnormal_dir = "data/abnormal_audio"
    model_save_path = "models/ocsvm.pkl"
    sr = 16000

    # 特征提取器
    feature_extractor = FeatureExtractor(sr=sr)

    # 加载正常音频
    normal_audios = load_audio_files(normal_dir, sr=sr)
    X_normal = extract_features_from_audios(normal_audios, feature_extractor)

    # 如果有异常数据，可用于验证，但 One-Class SVM 只需要正常数据
    # 训练模型
    print(f"Training on {len(X_normal)} normal samples...")
    model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)  # nu 为异常比例预期
    model.fit(X_normal)

    # 保存模型
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")

    # 可选：在异常数据上测试
    if os.path.exists(abnormal_dir):
        abnormal_audios = load_audio_files(abnormal_dir, sr=sr)
        X_abnormal = extract_features_from_audios(abnormal_audios, feature_extractor)
        normal_score = model.score_samples(X_normal).mean()
        abnormal_score = model.score_samples(X_abnormal).mean()
        print(f"Avg normal score: {normal_score:.3f}, avg abnormal score: {abnormal_score:.3f}")

if __name__ == "__main__":
    main()