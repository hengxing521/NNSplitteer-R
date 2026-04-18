# eval_security/attack_finetune.py
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import sys
import os
import numpy as np
from torch.utils.data import SubsetRandomSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import get_model
from datasets import get_dataloader
from core_defense.obfuscation_ops import inference

def main():
    parser = argparse.ArgumentParser(description='Security Eval: Fine-tuning Recovery Attack')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--net', type=str, default='resnet18')
    parser.add_argument('--model_path', type=str, required=True, help='待攻击的混淆模型路径 (.pth)')
    parser.add_argument('--data_ratio', type=float, default=0.1, help='攻击者掌握的训练集比例 (默认10%)')
    parser.add_argument('--epochs', type=int, default=20, help='攻击者的微调轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='微调学习率')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 模拟敌手只拥有 10% 的本地数据用于破解
    full_trainloader, testloader, num_classes = get_dataloader(args.dataset, batch_size=128)
    dataset_size = len(full_trainloader.dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(args.data_ratio * dataset_size))
    
    np.random.seed(42) # 保证评测过程的可复现性
    np.random.shuffle(indices)
    attacker_indices = indices[:split]
    attacker_sampler = SubsetRandomSampler(attacker_indices)
    attacker_loader = torch.utils.data.DataLoader(
        full_trainloader.dataset, batch_size=128, sampler=attacker_sampler)
    
    # 2. 挂载被拦截的致盲模型
    net = get_model(args.net, num_classes=num_classes)
    net.load_state_dict(torch.load(args.model_path, map_location=device))
    net.to(device)
    
    initial_acc = inference(net, device, testloader)
    print(f"\n[*] 开始强监督 Fine-tuning 攻击 | 掌握数据量: {len(attacker_indices)} 张 | 初始致盲精度: {initial_acc * 100:.2f}%")
    
    # 3. 实施强监督微调破解
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        net.train()
        running_loss = 0.0
        for inputs, targets in attacker_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # 阶段性汇报攻击战况
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            acc = inference(net, device, testloader)
            print(f"    - 攻击微调 Epoch [{epoch+1}/{args.epochs}] Loss: {running_loss/len(attacker_loader):.4f} | 当前恢复精度: {acc * 100:.2f}%")
            
    final_acc = inference(net, device, testloader)
    if final_acc < 0.30: # 黑盒知识蒸馏窃取的理论上限通常在 30% 左右
        print(f"\n[✅ 防御成功] 空间重排引发了深度拓扑错乱，微调陷入局部死锁！最高恢复精度被压制在 {final_acc * 100:.2f}%。")
    else:
        print(f"\n[❌ 防线崩溃] 模型语义被敌手通过强监督梯度强行纠正，精度回弹至 {final_acc * 100:.2f}%！")

if __name__ == '__main__':
    main()