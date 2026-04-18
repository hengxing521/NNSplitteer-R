# core_defense/controller_rnn.py
import torch  
import torch.nn as nn  
import torch.optim as optim  
import numpy as np  
from core_defense.trainer_engine import Trainer

class Controller_rnn(nn.Module):
    """
    马尔可夫决策过程 (MDP) 下的 RNN 策略梯度寻址智能体
    """
    def __init__(self, device, layer_list, k, embedding_dim=128, hidden_dim=256):
        super(Controller_rnn, self).__init__()
        self.device = device
        self.layer_list = layer_list  
        self.k = k 
        self.embedding_dim = embedding_dim  
        self.hidden_dim = hidden_dim  

        # 隐状态驱动的网络架构
        self.network = nn.Sequential(nn.Linear(self.embedding_dim, self.hidden_dim), nn.ReLU())
        self.decoders = nn.ModuleList([nn.Linear(self.hidden_dim, i) for i in self.layer_list])
        self.rnn = nn.RNN(self.embedding_dim, self.hidden_dim, 1)
        self.init_parameters()  

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.1, 0.1)
        for decoder in self.decoders:
            decoder.bias.data.fill_(0)

    def forward(self, input, h_t, step):
        x = self.network(input).unsqueeze(0)
        output, h_t = self.rnn(x, h_t)
        output = output.squeeze(0)
        # 边界截断，防止策略梯度计算出现 log(0) 溢出
        prob = torch.clamp(torch.softmax(self.decoders[step](output), dim=-1), min=1e-8)
        return prob, h_t

    def train_controller(self, net, trainloader, testloader, arg):
        """基于 REINFORCE 算法的策略评估与模型优化"""
        optimizer_rl = optim.Adam(self.parameters(), lr=arg.lr_rl)
        best = -1.0  
        cnt = 0
        record = []

        print(f"\n[启动 RL 自动化寻址] 在全网特征流中搜索核心命门参数...")
        for epoch in range(arg.num_epoch_rl):
            rewards = []
            log_probs = []

            for _ in range(arg.batch_size_rl):
                layer_filters = []
                actions_p = []
                x_t = torch.randn(1, self.embedding_dim).to(self.device)
                h_t = torch.zeros(1, 1, self.hidden_dim).to(self.device)

                for step in range(len(self.layer_list)):
                    prob, h_t = self(x_t, h_t, step)
                    # 论文创新：多项式无放回抽样机制 (Multinomial Sampling without Replacement)
                    action = torch.multinomial(prob, self.k, replacement=False)
                    action_p = torch.log(prob.squeeze(0)[action])
                    actions_p.append(torch.sum(action_p))
                    layer_filters.append(action.tolist())

                # 评估动作的回报
                acc_cnn, layer_modi, final_masks, final_state_dict = Trainer(
                    arg, layer_filters, net, trainloader, testloader, self.device
                )

                reward = -acc_cnn  # 准确率越低，破坏力越强，奖励越高
                rewards.append(reward)

                if reward >= best:
                    ori_state_dict = {k: v.cpu() for k, v in net.state_dict().items()}
                    final_state_dict_cpu = {k: v.cpu() for k, v in final_state_dict.items()}
                    record = [acc_cnn, layer_filters, layer_modi, final_masks, ori_state_dict, final_state_dict_cpu]
                    best = reward  
                    cnt = 0        
                else:
                    cnt += 1       

                log_probs.append(sum(actions_p))

            if cnt > arg.max_iter:  
                print(">>> 警告：策略已收敛，触发早停机制 (Early Stopping)。")
                break

            # 策略梯度公式：计算优势函数 (Advantage) 并反向传播
            if len(rewards) > 1 and np.std(rewards) > 0:
                b = np.mean(rewards)
                loss = sum([(r - b) * lp for r, lp in zip(rewards, log_probs)]) / len(rewards)
            else:
                loss = sum([-r * lp for r, lp in zip(rewards, log_probs)]) / len(rewards)
                
            loss.backward()  
            optimizer_rl.step()  
            print(f'RL Epoch [{epoch+1}/{arg.num_epoch_rl}] | Batch 平均奖励(精度损失): {np.mean(rewards):.4f} | 最佳记录: {best:.4f}')

        return record