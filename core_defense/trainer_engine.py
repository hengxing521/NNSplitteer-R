# core_defense/trainer_engine.py
import torch
import torch.nn as nn
import copy
from core_defense.obfuscation_ops import get_layer_filter_info, initialize_masks, apply_obfuscation, inference

def Trainer(arg, layer_filters, net, trainloader, testloader, device):
    """
    基于泰勒敏感度度量的动态掩码剪枝引擎 (对应论文 Algorithm 2)
    """
    criterion = nn.CrossEntropyLoss()
    ori_state_dict = copy.deepcopy(net.state_dict())
    
    target_acc = arg.target_acc 
    target_ratio = arg.target_ratio
    obf_method = arg.obf_method

    # 1. 计算目标轻量化预算
    total_model_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    budget_limit = int(total_model_params * target_ratio)
    
    selected_info = get_layer_filter_info(net, layer_filters)
    candidate_masks = initialize_masks(selected_info, device)
    
    # 2. 执行初次致盲测试
    net.load_state_dict(apply_obfuscation(ori_state_dict, selected_info, candidate_masks, method=obf_method))
    current_acc = inference(net, device, testloader)
    
    if current_acc > target_acc:
        return current_acc, [0]*len(layer_filters), candidate_masks, net.state_dict()

    best_masks = copy.deepcopy(candidate_masks)
    
    # 3. 泰勒动态剪枝微调循环 (极限压缩至目标比例)
    for step in range(1, 20):
        net.train()
        inputs, targets = next(iter(trainloader))
        inputs, targets = inputs.to(device), targets.to(device)
        
        net.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()  # 获取当前参数流形的逻辑梯度
        
        global_scores = []
        global_indices_map = []
        
        # 耦合计算每个混淆点的绝对敏感度 S
        for info in selected_info:
            name = info['name']
            m = candidate_masks[name]
            m_flat = m.view(-1)
            
            grad_tensor = dict(net.named_parameters())[name].grad
            if grad_tensor is None:
                continue
                
            grad_flat = grad_tensor.view(-1)
            w_ori_flat = ori_state_dict[name].view(-1).to(device)
            w_cur_flat = dict(net.named_parameters())[name].data.view(-1)
            
            # 核心理论公式: S = |Grad| * |W_ori - W_cur|
            importance = torch.abs(grad_flat) * torch.abs(w_ori_flat - w_cur_flat)
            
            active_indices = torch.nonzero(m_flat, as_tuple=False).squeeze()
            if active_indices.dim() == 0: active_indices = active_indices.unsqueeze(0)
            
            for idx in active_indices:
                global_scores.append(importance[idx].item())
                global_indices_map.append((name, idx.item()))
                
        current_total_active = len(global_scores)
        if current_total_active <= budget_limit:
            break
            
        # 末位淘汰：跨层大排名，丢弃最底层 20% 的冗余掩码
        num_prune = int(current_total_active * 0.20)
        if num_prune == 0: break
        
        global_scores_tensor = torch.tensor(global_scores)
        _, lowest_idx = torch.topk(global_scores_tensor, num_prune, largest=False)
        
        for idx in lowest_idx:
            layer_name, flat_idx = global_indices_map[idx.item()]
            candidate_masks[layer_name].view(-1)[flat_idx] = False
            
        # 闭环验证精度反弹
        net.load_state_dict(apply_obfuscation(ori_state_dict, selected_info, candidate_masks, method=obf_method))
        test_acc = inference(net, device, testloader)
        
        if test_acc <= target_acc:
            best_masks = copy.deepcopy(candidate_masks)
            current_acc = test_acc
        else:
            # 触及防线承重墙，触发回退并退出
            break

    # 4. 生成最终输出
    layer_modi = []
    for info in selected_info:
        m = best_masks[info['name']]
        layer_modi.append(m.sum().item())
        
    net.load_state_dict(apply_obfuscation(ori_state_dict, selected_info, best_masks, method=obf_method))
    final_obf_dict = copy.deepcopy(net.state_dict())
    
    return current_acc, layer_modi, best_masks, final_obf_dict