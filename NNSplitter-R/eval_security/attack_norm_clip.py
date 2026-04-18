# eval_security/attack_norm_clip.py
import torch
import argparse
import sys
import os

# 将项目根目录加入路径，方便独立运行
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import get_model
from datasets import get_dataloader
from core_defense.obfuscation_ops import inference

def norm_clipping_attack(net, threshold=3.0):
    """
    范数裁剪攻击 (Norm Clipping Attack)
    攻击者逻辑：计算每层参数的均值 mu 和标准差 std，强制截断游离于 [mu - t*std, mu + t*std] 之外的异常尖峰。
    """
    print(f"[*] 正在实施 Norm Clipping 攻击 (裁剪阈值: {threshold} sigma)...")
    clipped_params_count = 0
    total_params = 0

    with torch.no_grad():
        for name, param in net.named_parameters():
            if 'weight' in name and len(param.shape) > 1: # 主要针对具有空间维度的权重
                mu = param.data.mean()
                std = param.data.std()
                
                lower_bound = mu - threshold * std
                upper_bound = mu + threshold * std
                
                # 统计被裁剪的参数量
                outliers = (param.data < lower_bound) | (param.data > upper_bound)
                clipped_params_count += outliers.sum().item()
                total_params += param.numel()
                
                # 强制裁剪抹平异常点
                param.data = torch.clamp(param.data, min=lower_bound, max=upper_bound)
                
    print(f"    - 全网扫描参数总数: {total_params}")
    print(f"    - 成功截断/抹平的异常点数量: {clipped_params_count} 个")
    return net

def main():
    parser = argparse.ArgumentParser(description='Security Eval: Norm Clipping Attack')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--net', type=str, default='resnet18')
    parser.add_argument('--model_path', type=str, required=True, help='待攻击的致盲模型路径 (.pth)')
    parser.add_argument('--threshold', type=float, default=3.0, help='裁剪阈值 t (默认 3.0 sigma)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 加载数据与模型
    _, testloader, num_classes = get_dataloader(args.dataset, batch_size=256)
    net = get_model(args.net, num_classes=num_classes)
    
    # 2. 挂载被致盲的混淆模型
    print(f"\n[蓝军测试] 目标致盲模型: {args.model_path}")
    net.load_state_dict(torch.load(args.model_path, map_location=device))
    
    initial_acc = inference(net, device, testloader)
    print(f"[!] 攻击前 (致盲状态) 准确率: {initial_acc * 100:.2f}%")
    
    # 3. 实施裁剪攻击
    net = norm_clipping_attack(net, args.threshold)
    
    # 4. 评估攻击后模型是否回弹
    restored_acc = inference(net, device, testloader)
    print(f"[!] 攻击后 (试图恢复) 准确率: {restored_acc * 100:.2f}%")
    
    if restored_acc < 0.20: # 以 20% 作为防线是否被击穿的参考阈值
        print(f"\n[✅ 防御成功] NNSplitter-R 多重集恒定特性使 Clipping 攻击完全失效！精度被死死压制。")
    else:
        print(f"\n[❌ 防线崩溃] 敌手成功利用统计漏洞剔除了混淆噪点，恢复了模型逻辑！")

if __name__ == '__main__':
    main()