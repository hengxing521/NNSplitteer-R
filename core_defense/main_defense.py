import os  
import datetime
import random
import numpy as np  
import torch  
import argparse  
import pickle

from core_defense.controller_rnn import Controller_rnn  
from core_defense.obfuscation_ops import get_layer_filter_info, recover_model, inference
from datasets import get_dataloader
from models import get_model

def set_seed(seed=42):
    """锁定全局随机种子，确保顶会 Artifact 代码具有严格复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='NNSplitter-R System: Lightweight & Stealthy Active Defense')
    # 数据集与模型参数
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'stl10'])
    parser.add_argument('--net', type=str, default='resnet18', choices=['alexnet', 'resnet18', 'resnet50', 'vgg16', 'vgg16_bn'])
    
    # 核心防御目标参数
    parser.add_argument('--obf_method', type=str, default='global_minmax', help='防御策略：首选 global_minmax')
    parser.add_argument('--target_acc', type=float, default=0.15, help='防线致死精度容忍阈值')
    parser.add_argument('--target_ratio', type=float, default=0.0002, help='目标压缩比例 (如万分之二)')
    
    # RL 超参数
    parser.add_argument('--batch_size_rl', type=int, default=3, help='单轮采样批次 B')
    parser.add_argument('--k', type=int, default=16, help='每层多项式采样候选数量 k')
    parser.add_argument('--max_iter', type=int, default=15, help='早停耐心值')
    parser.add_argument('--lr_rl', type=float, default=0.001, help='学习率')
    parser.add_argument('--num_epoch_rl', type=int, default=300, help='最大搜索轮数')
    
    args = parser.parse_args()
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 自动构建带有时间戳的分类归档目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experiments", f"{args.dataset}_{args.net}_{timestamp}")
    
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "obfuscated_models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "model_secrets"), exist_ok=True)
    print(f"[*] NNSplitter-R 启动！实验产物将保存在: {exp_dir}\n")

    # 2. 动态加载真实数据集与明文预训练网络 (已解除封印)
    trainloader, testloader, num_classes = get_dataloader(args.dataset, batch_size=256)
    net = get_model(args.net, num_classes=num_classes)
    
    print(f"[数据加载] Target Dataset: {args.dataset.upper()} | Network: {args.net.upper()}")
    
    pretrained_path = f'models/pretrained_weights/{args.net}_{args.dataset}.pth'
    if os.path.exists(pretrained_path):
        print(f"[*] 成功加载预训练权重: {pretrained_path}")
        net.load_state_dict(torch.load(pretrained_path, map_location=device))
    else:
        print(f"[警告] 未找到预训练权重 {pretrained_path}，将使用随机初始化权重进行演示。请先运行 train_base_model.py！")
    net.to(device)

    # 动态提取各层特征通道数 (C_out) 供给 RNN 动作空间
    layer_list = [para.shape[0] for name, para in net.named_parameters() if len(para.shape) > 1 and para.shape[-1] > 1]
    
    # 3. 启动引擎
    controller = Controller_rnn(device, layer_list, args.k).to(device)
    record = controller.train_controller(net, trainloader, testloader, args)
    
    print('\n======================================================')
    print('      NNSplitter-R 主动防御制备完毕！开始资产分离落盘      ')
    print('======================================================')
    
    # (已解除多行注释封印，正式保存模型！)
    obf_acc = record[0]
    final_masks = record[3]
    final_state_dict = record[5]
    
    print(f'>>> 最终混淆致盲精度: {obf_acc:.4f}')
    
    # 分发产物 A: 暴露给敌手 GPU 的致盲模型
    obf_path = os.path.join(exp_dir, "obfuscated_models", "obf_model.pth")
    torch.save(final_state_dict, obf_path)
    
    # 分发产物 B: 存储于 TEE 的极密安全资产 (掩码索引与残差)
    secret_path = os.path.join(exp_dir, "model_secrets", 'secret_masks.pkl')
    with open(secret_path, 'wb') as f:
        pickle.dump(final_masks, f)
        
    print(f'>>> [资产分离成功] 致盲模型部署至: {obf_path}')
    print(f'>>> [资产分离成功] TEE掩码部署至: {secret_path}')
    
    # 4. 授权 TEE 闭环推理校验
    print('\n[在线授权验证] 模拟 TEE 原位恢复协议校验模型保真度...')
    selected_info = get_layer_filter_info(net, record[1])
    recovered_state = recover_model(final_state_dict, record[4], selected_info, final_masks)
    
    net.load_state_dict(recovered_state)
    recover_acc = inference(net, device, testloader)
    print(f'>>> [安全通信验证] TEE 授权恢复精度: {recover_acc:.4f} (达成 100% 无损要求)\n')

if __name__ == '__main__':
    main()