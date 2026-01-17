#!/bin/bash
# 🚀 Docker 版 AI 总控台 (Final Release)
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# 检查 Docker 是否运行
if ! docker ps > /dev/null 2>&1; then
    echo "❌ Docker 未运行！请先启动 Docker Desktop 或服务。"
    exit 1
fi

echo -e "${BLUE}=== 正在启动 AI 服务环境... ===${NC}"
# 启动容器
docker-compose up -d

# 定义容器内执行函数
run_in_toolbox() {
    # $1: 脚本路径 (相对于 scripts 目录)
    echo -e "${GREEN}🚀 正在容器内执行: $1 ...${NC}"
    # 使用 docker exec 调用容器内的 python
    docker-compose exec -it ai_toolbox python /app/scripts/$1
    echo -e "${BLUE}✅ 任务完成。按回车键继续...${NC}"
    read
}

while true; do
    clear
    echo -e "${BLUE}=======================================================${NC}"
    echo -e "${BLUE}       🚀 AI 智能标注交付系统 (Docker v3.0)${NC}"
    echo -e "${BLUE}=======================================================${NC}"
    
    echo "   1. 📊 打开 Label Studio (http://localhost:8080)"
    echo ""
    echo -e "${GREEN}[P1: 产品图片]${NC}"
    echo "   2. 📦 训练 (auto_yolo_manager.py)"
    echo "   3. 🖌️  推理 (yolo_to_ls.py)"
    echo ""
    echo -e "${GREEN}[P4: 视频画面]${NC}"
    echo "   4. 🎬 训练 (auto_video_yolo.py)"
    echo "   5. 🖌️  推理 (yolo_to_ls.py)"
    echo ""
    echo -e "${GREEN}[P2: 纯音频]${NC}"
    echo "   6. 📦 训练 (auto_train_manager.py)"
    echo "   7. 🎧 推理 (whisper_to_ls.py)"
    echo ""
    echo -e "${GREEN}[P3: 视频语音]${NC}"
    echo "   8. 🎬 训练 (auto_video_whisper.py)"
    echo "   9. 🎧 推理 (whisper_to_ls.py)"
    echo ""
    echo -e "${GREEN}[P8: 目标追踪]${NC}"
    echo "   10. ⚡ 自动追踪 (auto_tracker.py)"
    echo ""
    echo "   q. 退出"
    
    read -p "👉 请选择: " choice

    case $choice in
        1) 
            echo "👉 请在浏览器访问: http://localhost:8080"
            echo "   (账号: admin@example.com / 密码: password123)"
            read 
            ;;
        # 👇 这里的路径已适配您的新目录名 (yolo_workspace)
        2) run_in_toolbox "yolo_workspace/auto_yolo_manager.py --project_id 1" ;;
        3) run_in_toolbox "yolo_to_ls.py --project 1" ;;
        
        4) run_in_toolbox "train_yolo_video/auto_video_yolo.py" ;;
        5) run_in_toolbox "yolo_to_ls.py --project 4" ;;
        
        6) run_in_toolbox "whisper_workspace/auto_train_manager.py" ;;
        7) run_in_toolbox "whisper_to_ls.py --project 2" ;;
        
        8) run_in_toolbox "train_whisper_video/auto_video_whisper.py" ;;
        9) run_in_toolbox "whisper_to_ls.py --project 3" ;;
        
        10) run_in_toolbox "video_tracking_workspace/auto_tracker.py" ;;
        
        q) exit 0 ;;
        *) echo "❌ 无效选择"; sleep 1 ;;
    esac
done
