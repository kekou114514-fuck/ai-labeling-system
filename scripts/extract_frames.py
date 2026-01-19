import cv2
import os
import sys
import glob

# 实时日志输出
sys.stdout.reconfigure(line_buffering=True)

# === 配置路径 (与 Docker 环境一致) ===
DATA_ROOT = os.getenv('DATA_ROOT', '/data')
VIDEO_DIR = os.path.join(DATA_ROOT, 'videos')
FRAME_DIR = os.path.join(DATA_ROOT, 'video_frames')

# 抽帧间隔：每隔 10 帧提取一张 (约 3fps)
FRAME_INTERVAL = 10 

def extract():
    # 1. 检查视频目录
    if not os.path.exists(VIDEO_DIR):
        print(f"❌ 错误：找不到视频目录 {VIDEO_DIR}")
        print("💡 请将视频文件 (.mp4, .avi) 放入宿主机的 project_data/videos 目录")
        return
    
    os.makedirs(FRAME_DIR, exist_ok=True)
    
    # 2. 扫描视频 (不区分大小写)
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    videos = []
    for ext in video_extensions:
        videos.extend(glob.glob(os.path.join(VIDEO_DIR, ext)))
        videos.extend(glob.glob(os.path.join(VIDEO_DIR, ext.upper())))
    
    if not videos:
        print(f"❓ 在 {VIDEO_DIR} 中没有发现视频文件。")
        return

    print(f"🚀 准备处理 {len(videos)} 个视频...")

    # 3. 开始抽帧
    for v_path in videos:
        v_name = os.path.basename(v_path)
        v_prefix = os.path.splitext(v_name)[0]
        
        cap = cv2.VideoCapture(v_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 正在处理: {v_name} (共 {total_frames} 帧)")
        
        count = 0
        saved_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if count % FRAME_INTERVAL == 0:
                # 文件名格式: 视频名_帧号.jpg
                frame_name = f"{v_prefix}_f{count:06d}.jpg"
                save_path = os.path.join(FRAME_DIR, frame_name)
                cv2.imwrite(save_path, frame)
                saved_count += 1
            
            count += 1
            
        cap.release()
        print(f"   ✅ 提取了 {saved_count} 张图片")

    print(f"🎉 所有视频处理完毕。图片保存在: {FRAME_DIR}")
    print("💡 下一步：请去 Label Studio Project 4 点击 'Sync' 按钮同步图片！")

if __name__ == "__main__":
    extract()
