# core_defense/obfuscation_ops.py
import torch
import copy

def get_layer_filter_info(model, layer_list):
    """提取模型结构信息，归档被 RNN 选中的滤波器索引"""
    selected_info = []
    cnt = 0
    for name, para in model.named_parameters():
        # 仅针对具有空间特征映射的层（如 Conv2d 和 Linear）
        if len(para.shape) > 1 and para.shape[-1] > 1:
            if cnt < len(layer_list):
                filter_indices = layer_list[cnt]
                selected_info.append({
                    'name': name,
                    'filter_indices': filter_indices,
                    'shape': para.shape,
                    'numel': para.numel()
                })
            cnt += 1
    return selected_info

def initialize_masks(selected_info, device):
    """初始化用于指示重排坐标的布尔型掩码张量 M"""
    masks = {}
    for info in selected_info:
        name = info['name']
        f_indices = info['filter_indices']
        m = torch.zeros(info['shape'], dtype=torch.bool, device=device)
        for f_idx in f_indices:
            m[f_idx] = True  # 标记活跃的滤波器区域
        masks[name] = m
    return masks

def apply_obfuscation(ori_state_dict, selected_info, masks, method='global_minmax'):
    """
    [核心防御算子] 全局空间重排引擎 (对应论文 Algorithm 1)
    严格保证多重集恒定 (Multiset Invariance)，绝对不引入外部新数值。
    """
    new_dict = copy.deepcopy(ori_state_dict)
    
    if method == 'global_minmax':
        global_active_params = []
        
        # 1. 收集全网所有被掩码标记的活跃参数
        for info in selected_info:
            name = info['name']
            m_flat = masks[name].view(-1)
            if m_flat.sum() > 0:
                active_vals = ori_state_dict[name].flatten()[m_flat]
                global_active_params.append(active_vals)
                
        if len(global_active_params) == 0:
            return new_dict
            
        # 2. 拼接全局参数池，执行极值首尾反转 (制造最大语义错位)
        v_global = torch.cat(global_active_params)
        v_sorted, idx_sorted = torch.sort(v_global, descending=False)
        v_reversed = torch.flip(v_sorted, dims=[0])
        
        v_new = torch.zeros_like(v_global)
        v_new[idx_sorted] = v_reversed
        
        # 3. 依据原始索引映射，原位回填反转后的数值
        ptr = 0
        for info in selected_info:
            name = info['name']
            m_flat = masks[name].view(-1)
            num_active = m_flat.sum().item()
            if num_active > 0:
                w_new_flat = ori_state_dict[name].flatten().clone()
                w_new_flat[m_flat] = v_new[ptr : ptr + num_active]
                new_dict[name] = w_new_flat.view(info['shape'])
                ptr += num_active
                
    return new_dict

def recover_model(obfuscated_dict, ori_state_dict, selected_info, masks):
    """
    [TEE 原位复原协议]
    利用保存在 TEE 内的机密索引，执行极低开销的指针重定向覆盖，实现 100% 零浮点误差还原。
    """
    recovered_dict = copy.deepcopy(obfuscated_dict)
    for info in selected_info:
        name = info['name']
        m = masks[name]
        m_flat = m.view(-1)
        if m_flat.sum() > 0:
            w_ori_flat = ori_state_dict[name].flatten()
            w_rec_flat = recovered_dict[name].flatten()
            # 仅将发生过重排的位置覆写为原值
            w_rec_flat[m_flat] = w_ori_flat[m_flat]
            recovered_dict[name] = w_rec_flat.view(info['shape'])
    return recovered_dict

def inference(net, device, testloader):
    """标准测试集前向推理评估"""
    net.to(device)
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total