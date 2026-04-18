# NNSplitter-R: 基于空间重排与泰勒剪枝的轻量级高隐蔽模型主动防御系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Artifact Evaluation](https://img.shields.io/badge/Artifact-Evaluated-brightgreen.svg)]()

本仓库是学术论文 **"NNSplitter-R: Lightweight and Stealthy Active Defense for DNNs via Spatial Rearrangement and Taylor Pruning"** 的官方 PyTorch 实现代码。

## 💡 项目简介 (Overview)
随着深度学习模型向边缘设备的广泛部署，暴露在不可信硬件环境（如不可信 GPU/OS）中的高价值模型面临着严重的白盒提取与窃取威胁。现有的基于 TEE（可信执行环境）的轻量级主动防御方案（如 NNSplitter, Magnitude）普遍受制于**“随机性脆弱”**缺陷：通过引入外源数值来篡改参数，会在统计学上产生可被 KDE（核密度估计）和范数裁剪探测的“概率密度尖峰”，导致防线被轻易击穿。

**NNSplitter-R** 提出了一种全新的代数免疫路径，包含三大核心机制：
1. **全局极值首尾置换算法 (Global Min-Max Rearrangement)**：仅通过交换参数的物理坐标映射来阻断特征流，在数学上保证全网权重**多重集（Multiset）的绝对恒定**。KDE 分布完美保真，从根源上使统计学恢复攻击失效。
2. **泰勒一阶动态剪枝 (Dynamic Taylor Pruning)**：耦合了逻辑梯度与物理位移偏差（$S = |\nabla W| \cdot |\Delta W|$），将 TEE 需要保护的机密存储负荷极限压缩至全网参数量的 **0.0002% (万分之二)**，突破了轻量化防御的边界。
3. **极低开销的原位复原协议 (Zero-error In-place Restoration)**：在授权 TEE 中，仅需稀疏指针重定向，无任何浮点计算误差。

---

## 🗂️ 代码库结构 (Repository Structure)
本项目遵循顶会代码产物规范，将核心逻辑、数据集、模型架构与蓝军评估已完全模块化解耦：

```text
NNSplitter-R/
│
├── README.md                  # [核心门面] 项目介绍、环境依赖、一键复现指令、论文引用
├── requirements.txt           # [环境配置] torch, torchvision, numpy, scipy, matplotlib等
├── LICENSE                    # [开源协议] 建议使用 MIT 或 Apache 2.0
│
├── ⚙️ core_defense/             # 【模块一：核心混淆与防御引擎】(整合你上传的4个源码)
│   ├── main_defense.py        # 🚀 统一主入口：解析参数(--net, --dataset)，串联RL与剪枝
│   ├── controller_rnn.py      # RNN 寻址智能体 (隐向量驱动、策略梯度更新、多项式采样)
│   ├── trainer_engine.py      # 泰勒一阶敏感度评估与动态掩码迭代剪枝 (原 train.py)
│   └── obfuscation_ops.py     # 空间重排算子(Global Min-Max)与TEE无损恢复(原 utils.py)
│
├── 📦 datasets/                 # 【模块二：数据集加载器】(支持不同分辨率与预处理)
│   ├── __init__.py            # 统一的数据路由工厂函数
│   ├── cifar.py               # CIFAR-10 / CIFAR-100 加载与数据增强 (32x32)
│   └── stl10.py               # STL-10 加载与中心裁剪预处理 (96x96 大感受野)
│
├── 🧠 models/                   # 【模块三：目标网络拓扑定义】(分离架构定义，方便统一接口)
│   ├── __init__.py            # 模型路由工厂，根据字符串动态实例化网络
│   ├── alexnet.py             # AlexNet 适配代码 (浅层大卷积核测试)
│   ├── resnet.py              # ResNet-18 / ResNet-50 适配代码 (深层残差拓扑测试)
│   ├── vgg.py                 # VGG-16_bn 适配代码 (直筒型深层拓扑测试)
│   └── pretrained_weights/    # 存放上述模型在3个数据集上的原始明文预训练权重 (.pth)
│
├── 🛡️ eval_security/            # 【模块四：蓝军评估与攻击模拟】(提纯自 GroupCover 框架)
│   ├── __init__.py            
│   ├── attack_norm_clip.py    # 统计攻击模拟：实施范数裁剪，检测异常分布尖峰
│   ├── attack_finetune.py     # 白盒微调模拟：使用 10% 辅助数据集强行恢复错乱拓扑
│   └── plot_kde_fidelity.py   # 统计学证明工具：读取权重并绘制 KDE 概率密度对比图
│
├── 📜 scripts/                  # 【模块五：一键复现 Shell 脚本】(极大提升审稿人体验)
│   ├── run_defense_c10_r18.sh # 一键运行：在 CIFAR-10 上对 ResNet-18 实施混淆与压缩
│   ├── run_defense_stl10.sh   # 一键运行：在 STL-10 高分辨率数据集上测试泛化性
│   └── run_security_eval.sh   # 一键运行：对生成的混淆模型自动执行裁剪与微调攻击
│
└── 📊 experiments/              # 【模块六：实验产物与日志归档】(由代码自动生成)
    ├── logs/                  # 存放 TensorBoard 或纯文本的 RL 奖励曲线日志
    ├── obfuscated_models/     # 存放处理后的致盲模型 (.pth，模拟部署于不可信 GPU)
    ├── model_secrets/         # 存放提取的机密字典 (.pkl，模拟部署于 TEE 机密内存)
    └── figures/               # 自动保存的精度对比条形图、KDE 曲线 PDF/PNG 原图