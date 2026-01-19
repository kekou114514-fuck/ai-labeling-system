import os
import subprocess
import sys

# 实时日志
sys.stdout.reconfigure(line_buffering=True)

# ==========================================
# ⚙️ 路径配置 (适配 P3 视频语音)
# ==========================================
DATA_ROOT = os.getenv('DATA_ROOT', '/data')
VIDEO_DIR = os.path.join(DATA_ROOT, "videos")
AUDIO_DIR = os.path.join(DATA_ROOT, "video_audio")

def extract_audio():
    # 1. 检查目录
    if not os.path.exists(VIDEO_DIR):
        print(f"❌ 找不到视频目录: {VIDEO_DIR}")
        return
    if not os.path.exists(AUDIO_DIR):
        os.makedirs(AUDIO_DIR, exist_ok=True)
        print(f"📁 已创建音频输出目录: {AUDIO_DIR}")

    # 2. 扫描视频
    video_extensions = ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv')
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(video_extensions)]
    
    if not video_files:
        print(f"❌ 在 {VIDEO_DIR} 中未找到视频文件。")
        return

    print(f"🎬 发现 {len(video_files)} 个视频，准备提取音频...")

    # 3. 提取音频 (16kHz 单声道 -> 适配 Whisper)
    count = 0
    for video_name in video_files:
        video_path = os.path.join(VIDEO_DIR, video_name)
        audio_name = os.path.splitext(video_name)[0] + ".wav"
        audio_path = os.path.join(AUDIO_DIR, audio_name)

        if os.path.exists(audio_path):
            # print(f"⏭️  跳过已存在: {audio_name}")
            continue

        print(f"🎤 正在提取: {video_name} -> {audio_name}")
        
        cmd = [
            "ffmpeg", "-i", video_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            audio_path, "-y", "-loglevel", "error"
        ]

        try:
            subprocess.run(cmd, check=True)
            count += 1
        except Exception as e:
            print(f"❌ 提取失败: {e}")

    print(f"\n✅ 提取完成！本次处理 {count} 个文件。")
    print(f"📂 音频存放于: {AUDIO_DIR}")
    print("💡 请去 Label Studio 项目 5 点击 'Settings -> Cloud Storage -> Sync' 同步数据。")

if __name__ == "__main__":
    extract_audio()
