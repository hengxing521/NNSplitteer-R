#!/bin/bash
# 一键运行：对生成的混淆模型自动执行裁剪与微调攻击

if [ -z "$1" ] || [ -z "$2" ]; then
  echo ">>> 错误：参数缺失！"
  echo ">>> 使用方法: bash scripts/run_security_eval.sh <待攻击混淆模型路径.pth> <对应的明文原模型路径.pth>"
  echo ">>> 示例: bash scripts/run_security_eval.sh experiments/.../obfuscated_models/obf_model.pth models/pretrained_weights/resnet18_cifar10.pth"
  exit 1
fi

TARGET_MODEL=$1
ORI_MODEL=$2

echo "========================================================="
echo "  [安全评估蓝军测试] 目标致盲模型: $TARGET_MODEL"
echo "========================================================="

echo -e "\n>>> [1/3] 正在执行: 范数裁剪攻击 (Norm Clipping)..."
python eval_security/attack_norm_clip.py --model_path "$TARGET_MODEL" --threshold 3.0

echo -e "\n>>> [2/3] 正在执行: 强监督微调恢复攻击 (Fine-tuning)..."
# 模拟敌手掌握 10% 数据，进行 10 个 epoch 的强制微调试图恢复语义
python eval_security/attack_finetune.py --model_path "$TARGET_MODEL" --epochs 10

echo -e "\n>>> [3/3] 正在生成: KDE 分布概率密度保真度验证图..."
python eval_security/plot_kde_fidelity.py \
    --ori_model "$ORI_MODEL" \
    --obf_model "$TARGET_MODEL"

echo -e "\n[*] 所有蓝军安全测试执行完毕！KDE 图像已保存至 experiments/figures/ 目录中。"