import cv2
import os
import subprocess
from pathlib import Path
from typing import List, Optional
import tempfile


def extract_frames(video_path: str, interval_seconds: float = 1) -> List:
    """
    Trích xuất frames từ video theo khoảng thời gian.
    
    Args:
        video_path: Đường dẫn đến file video
        interval_seconds: Khoảng thời gian giữa các frames (giây)
    
    Returns:
        List các frames (numpy arrays)
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Không thể mở video: {video_path}")
        return frames
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"Không thể lấy FPS từ video: {video_path}")
        cap.release()
        return frames
    
    frame_interval = int(fps * interval_seconds)
    if frame_interval == 0:
        frame_interval = 1
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frames.append(frame)
        
        frame_count += 1
    
    cap.release()
    print(f"Tổng số frames trích xuất: {len(frames)}")
    return frames


def extract_audio(video_path: str, output_path: Optional[str] = None) -> str:
    """
    Tách audio từ video và lưu thành file WAV.
    
    Args:
        video_path: Đường dẫn đến file video
        output_path: Đường dẫn file audio output (nếu None sẽ tạo temp file)
    
    Returns:
        Đường dẫn đến file audio đã tách
    """
    if output_path is None:
        # Tạo temp file với extension .wav
        temp_dir = tempfile.gettempdir()
        video_name = Path(video_path).stem
        output_path = os.path.join(temp_dir, f"{video_name}_audio.wav")
    
    # Đảm bảo thư mục output tồn tại
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Sử dụng ffmpeg để tách audio
    try:
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # Không copy video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '16000',  # Sample rate 16kHz (phù hợp cho speech recognition)
            '-ac', '1',  # Mono
            '-y',  # Overwrite output file
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        
        print(f"Đã tách audio thành công: {output_path}")
        return output_path
        
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi tách audio: {e.stderr.decode()}")
        raise
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy ffmpeg. Vui lòng cài đặt ffmpeg.")
        raise


def is_video_file(file_path: str) -> bool:
    """Kiểm tra xem file có phải là video không"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
    return Path(file_path).suffix.lower() in video_extensions

