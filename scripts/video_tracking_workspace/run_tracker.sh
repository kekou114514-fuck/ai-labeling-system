#!/bin/bash
echo "=========================================="
echo "   🚀 视频自动追踪系统 (Project ID: 6)"
echo "=========================================="
echo "1. 📹 自动追踪 (生成预标注)"
echo "2. 📦 闭环训练 (导出数据 -> 微调)"
echo ""
read -p "👉 请选择: " choice

if [ "$choice" == "1" ]; then
    python3 /app/scripts/video_tracking_workspace/auto_tracker.py
elif [ "$choice" == "2" ]; then
    read -p "👉 请输入项目 ID (默认 6): " pid
    pid=${pid:-6}
    python3 /app/scripts/video_tracking_workspace/auto_video_tracker_train.py --project_id $pid
else
    echo "❌ 无效选择"
fi
