#!/bin/bash
# 一键运行：在 CIFAR-10 上对 ResNet-18 实施混淆与压缩

echo "========================================================="
echo "  [NNSplitter-R] 启动防御生成: ResNet-18 on CIFAR-10     "
echo "========================================================="

# 调用核心防御引擎，使用本文首创的 global_minmax 策略
python core_defense/main_defense.py \
    --dataset cifar10 \
    --net resnet18 \
    --obf_method global_minmax \
    --target_acc 0.15 \
    --target_ratio 0.0002 \
    --num_epoch_rl 200

echo ""
echo "[*] CIFAR-10 上的 ResNet-18 混淆与压缩执行完毕！"
echo "[*] 致盲模型与 TEE 机密密钥已自动存入 experiments/ 对应时间戳目录中。"