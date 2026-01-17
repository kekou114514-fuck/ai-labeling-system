📘 AI 智能标注交付系统操作手册 (Docker 版)

版本: v3.0 最后更新: 2026-01-17
1. 系统简介

本系统基于 Docker 容器化技术，集成了 Label Studio（数据标注平台）与 AI 智能算法引擎（YOLO视觉模型 + Whisper语音模型）。支持图像检测、视频抽帧、语音转写及视频目标追踪等多种任务的自动化预标注与模型微调。
2. 目录结构说明

在开始使用前，请确保文件存放在正确的目录下。系统会自动读取 project_data 中的数据。
Plaintext

AI_Labeling_System/
├── run_host.sh          # [核心] 总控制启动脚本
├── env                  # [配置] 存放 API Key
├── docker-compose.yml   # [核心] 容器配置文件
├── models/              # [模型] 存放离线模型 (yolov8n.pt, whisper等)
└── project_data/        # [数据] 用户数据存放区 (请严格按以下分类存放)
    ├── images/          # 🖼️ P1: 产品图片 (放 .jpg, .png)
    ├── video_frames/    # 🎬 P4: 视频抽帧图片 (放 .jpg)
    ├── audio/           # 🎧 P2: 纯音频文件 (放 .wav, .mp3)
    ├── video_audio/     # 🎬 P3: 视频提取的音频 (放 .wav)
    ├── videos/          # 📹 P8: 原始视频文件 (放 .mp4)
    └── outputs/         # 📤 [自动生成] 算法推理生成的 JSON 结果

3. 首次部署与配置 (仅第一次需要)
3.1 启动系统

在项目根目录下打开终端，运行：
Bash

./run_host.sh

如果是首次运行，系统会自动下载所需镜像，请耐心等待直到出现菜单界面。
3.2 获取并配置 API Key

由于 Label Studio 需要初始化才能生成 Key，请按以下顺序操作：

    保持脚本运行，打开浏览器访问：http://localhost:8080

    注册/登录账号：

        默认账号: admin@example.com

        默认密码: password123

    点击右上角头像 -> Account & Settings。

    找到 Access Token，点击右侧的 Copy 按钮复制 Key。

    回到项目文件夹，打开 env 文件，填入 Key：
    Ini, TOML

    LS_API_KEY=你的真实Key粘贴在这里

    重载配置：

        在 run_host.sh 的菜单中选择 11 (如有)；

        或者新建终端运行：docker-compose restart ai_toolbox。

4. 日常使用指南
启动方式

每次使用前，只需运行：
Bash

./run_host.sh

🎯 模块操作流程
[P1] 产品图片检测 (Images)

    准备数据：将图片放入 project_data/images/

    功能 A：自动训练

        在 Label Studio 中完成部分标注并点击 Export。

        在脚本菜单选择 2 (训练)。

        系统会自动拉取标注数据 -> 转换格式 -> 微调 YOLO 模型。

    功能 B：AI 预标注 (推理)

        在脚本菜单选择 3 (推理)。

        等待提示“生成完毕”。

        前往 Label Studio -> 项目 P1 -> Import -> 上传 project_data/outputs/pre_annotations_images.json。

[P4] 视频画面检测 (Video Frames)

    准备数据：将视频抽帧后的图片放入 project_data/video_frames/

    操作：

        训练：菜单选择 4。

        推理：菜单选择 5 (结果生成在 outputs/pre_annotations_video_frames.json)。

[P2] 纯音频转写 (Audio)

    准备数据：将音频放入 project_data/audio/

    操作：

        训练：菜单选择 6 (需先在 LS 中标注并导出)。

        推理：菜单选择 7 (自动识别音频内容，生成 JSON)。

        导入：将生成的 pre_annotations_audio.json 导入 Label Studio P2 项目。

[P3] 视频语音转写 (Video Audio)

    准备数据：将从视频提取的音频放入 project_data/video_audio/

    操作：

        训练：菜单选择 8。

        推理：菜单选择 9。

[P8] 目标自动追踪 (Video Tracking)

    准备数据：将 MP4 视频放入 project_data/videos/

    操作：

        菜单选择 10。

        系统会自动对视频中的物体进行追踪，并生成轨迹 JSON。

        生成的 track_xxx.json 可直接导入 Label Studio 的视频项目。

5. 常见问题排查 (Troubleshooting)
Q1: 运行脚本提示 "Docker 未运行" 或 "Permission denied"？

原因：当前用户没有 Docker 权限。 解决：
Bash

sudo usermod -aG docker $USER
newgrp docker

Q2: 浏览器打不开 localhost:8080？

原因：容器正在启动中，或者数据库初始化未完成。 解决：

    请等待 1-2 分钟后再刷新。

    检查容器状态：docker ps -a。如果状态是 Exited，请联系技术支持查看日志。

Q3: 训练或推理时提示“找不到文件”？

解决：

    请严格按照第 2 节的目录结构存放文件。

    确保文件名不包含特殊字符（建议使用英文和数字）。

    检查 project_data 文件夹是否有读取权限 (sudo chmod -R 777 project_data)。

Q4: 如何查看系统日志？

如果程序卡住或报错，可以查看后台日志：
Bash

# 查看 AI 工具箱日志
docker logs -f ai_toolbox_worker

# 查看 Label Studio 日志
docker logs -f label_studio_server

6. 系统维护

    完全停止系统：
    Bash

    docker-compose down

    清理缓存 (慎用)： 如果需要彻底重置，删除 ls_data 文件夹（警告：会丢失所有标注记录）和 project_data 下的内容。

技术支持：如有其他报错，请截图终端的红色报错信息进行反馈。
