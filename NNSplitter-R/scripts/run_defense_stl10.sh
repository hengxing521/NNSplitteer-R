#!/bin/bash
# 一键运行：在 STL-10 高分辨率数据集上测试泛化性

echo "========================================================="
echo "  [NNSplitter-R] 启动泛化测试: ResNet-50 on STL-10 (96x96)"
echo "========================================================="

python core_defense/main_defense.py \
    --dataset stl10 \
    --net resnet50 \
    --obf_method global_minmax \
    --target_acc 0.15 \
    --target_ratio 0.0002 \
    --num_epoch_rl 150

echo ""
echo "[*] STL-10 高分辨率泛化防御测试完毕！"
echo "[*] 产物已自动存入 experiments/ 对应时间戳目录中。"