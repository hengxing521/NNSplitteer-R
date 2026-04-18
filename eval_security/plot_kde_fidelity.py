# eval_security/plot_kde_fidelity.py
import torch
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import get_model

def plot_kde_comparison(ori_weights, obf_weights, save_path):
    """利用 Seaborn 绘制原始权重与混淆权重的核密度估计(KDE)平滑曲线"""
    plt.figure(figsize=(10, 6))
    
    print("[*] 正在拟合原模型 KDE 曲线 (红线)...")
    sns.kdeplot(ori_weights, color="red", label="Original Model (Plaintext)", linewidth=2.5, linestyle="-")
    
    print("[*] 正在拟合混淆模型 KDE 曲线 (蓝虚线)...")
    sns.kdeplot(obf_weights, color="blue", label="Obfuscated Model (NNSplitter-R)", linewidth=2.5, linestyle="--")
    
    plt.title("Kernel Density Estimation (KDE) of Weight Distribution", fontsize=16, fontweight='bold')
    plt.xlabel("Weight Value", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 限制 X 轴显示范围，过滤掉极端的游离点，聚焦核心钟形分布区
    plt.xlim(np.percentile(ori_weights, 0.1), np.percentile(ori_weights, 99.9))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[✅ 成功] KDE 分布对比图已保存至: {save_path}")
    print("    -> 【论文检查点】请观察图表：红蓝两条曲线应当完美重合，证明多重集绝对恒定！")

def extract_layer_weights(state_dict, target_layer=None):
    """将目标层或全网的权重张量展平为 1D Numpy 数组，用于统计学分析"""
    weights = []
    for name, param in state_dict.items():
        if 'weight' in name and len(param.shape) > 1:
            if target_layer is None or target_layer in name:
                weights.append(param.cpu().numpy().flatten())
    return np.concatenate(weights)

def main():
    parser = argparse.ArgumentParser(description='Security Eval: KDE Fidelity Visualizer')
    parser.add_argument('--net', type=str, default='resnet18')
    parser.add_argument('--classes', type=int, default=10)
    parser.add_argument('--ori_model', type=str, required=True, help='原生明文模型权重路径 (.pth)')
    parser.add_argument('--obf_model', type=str, required=True, help='被混淆保护的模型权重路径 (.pth)')
    parser.add_argument('--target_layer', type=str, default=None, help='可指定分析特定层(如 layer1.0.conv1), 默认分析全网')
    parser.add_argument('--save_dir', type=str, default='experiments/figures')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"kde_fidelity_{args.net}.png")

    print(f"\n[*] 正在读取模型权重档案进行统计学比对...")
    ori_state = torch.load(args.ori_model, map_location='cpu')
    obf_state = torch.load(args.obf_model, map_location='cpu')
    
    ori_w = extract_layer_weights(ori_state, args.target_layer)
    obf_w = extract_layer_weights(obf_state, args.target_layer)
    
    # 为了绘图引擎效率，在全网参数中随机均匀采样 10 万个点进行 KDE 拟合已足以反映全局特征
    sample_size = min(100000, len(ori_w))
    np.random.seed(42)
    ori_w_sample = np.random.choice(ori_w, sample_size, replace=False)
    obf_w_sample = np.random.choice(obf_w, sample_size, replace=False)

    plot_kde_comparison(ori_w_sample, obf_w_sample, save_path)

if __name__ == '__main__':
    main()