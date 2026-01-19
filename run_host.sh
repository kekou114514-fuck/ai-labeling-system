#!/bin/bash
# 🚀 Docker 版 AI 总控台 (v4.5 全功能版)
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

if ! docker ps > /dev/null 2>&1; then
    echo "❌ Docker 未运行！"
    exit 1
fi

docker-compose up -d

run_in_toolbox() {
    docker exec -it ai_toolbox_worker python3 /app/scripts/$1
}

run_shell_in_toolbox() {
    docker exec -it ai_toolbox_worker /bin/bash /app/scripts/$1
}

ask_id() {
    local default_id=$1
    echo -e "${YELLOW}💡 确认 URL ID${NC}" >&2
    read -p "👉 ID (默认 $default_id): " pid
    echo "${pid:-$default_id}"
}

while true; do
    clear
    echo -e "${BLUE}=== 🚀 AI 智能标注系统 (v4.5) ===${NC}"
    echo "   1. 📊 打开 Label Studio"
    echo ""
    echo -e "${GREEN}[P1: 图片检测]${NC}"
    echo "   2. 训练 (ID:1) | 3. 推理 (ID:1)"
    echo -e "${GREEN}[P4: 视频抽帧]${NC}"
    echo "   4. 抽帧 | 5. 训练 (ID:4) | 6. 推理 (ID:4)"
    echo -e "${GREEN}[P3: 视频语音]${NC}"
    echo "   9. 提音频 | 10. 训练 (ID:5) | 11. 推理 (ID:5)"
    echo -e "${GREEN}[P8: 目标追踪 (ID: 6)]${NC}"  # 🔥 新增
    echo "   12. 📹 自动追踪与训练 (run_tracker.sh)"
    echo ""
    echo "   q. 退出"
    read -p "👉 选择: " choice

    case $choice in
        1) echo "访问 http://localhost:8080 (admin@example.com / password123)"; read ;;
        
        2) pid=$(ask_id 1); run_in_toolbox "yolo_workspace/auto_yolo_manager.py --project_id $pid" ;;
        3) pid=$(ask_id 1); run_in_toolbox "yolo_to_ls.py --project $pid" ;;
        
        4) run_in_toolbox "extract_frames.py" ;;
        5) pid=$(ask_id 4); run_in_toolbox "train_yolo_video/auto_video_yolo.py --project_id $pid" ;;
        6) pid=$(ask_id 4); run_in_toolbox "yolo_to_ls.py --project $pid" ;;
        
        9) run_in_toolbox "extract_audio.py" ;;
        10) pid=$(ask_id 5); run_in_toolbox "train_whisper_video/auto_video_whisper.py --project_id $pid" ;;
        11) pid=$(ask_id 5); run_in_toolbox "whisper_to_ls.py --project $pid" ;;
        
        12) run_shell_in_toolbox "video_tracking_workspace/run_tracker.sh" ;; # 🔥 新功能
        
        q) exit 0 ;;
        *) echo "❌ 无效"; read ;;
    esac
    echo -e "${BLUE}✅ 完成。按回车继续...${NC}"; read
done
