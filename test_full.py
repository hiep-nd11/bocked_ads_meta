import cv2
import base64
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
from pathlib import Path
from typing import List, Dict

# ===========================
# CONFIG
# ===========================
API_URL = "http://162.213.119.141:40484/v1/chat/completions"
MODEL_NAME = "vlm-7b"
API_KEY = "mysecretkey123"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}

# ===========================
# PROMPT TEMPLATE
# ===========================
PROMPT_TEMPLATE = """
Act as a strict Meta (Facebook/Instagram) Advertising Policy compliance expert.

Analyze the attached images. These images are intended to be used as ad creatives.
Your task is to identify potential policy violations based on Meta's Advertising Standards.

Focus specifically on the following policies:
1. Adult Content & Sexual Suggestiveness: 
   - Check for nudity, implied nudity, or excessive visible skin.
   - Check for sexually suggestive poses (e.g., arching back, lying on a bed in a provocative manner).
   - Check for images that focus unnecessarily on specific body parts (zoom-ins on skin/body).
2. Sensational Content: Are there images that might be considered shocking, scary, or gruesome (e.g., looking through body parts)?
3. Low Quality or Disruptive Content.
4. Prohibited Loan & Financial Services Content:
   - Reject any images that display, suggest, or imply loan services, money lending, credit offers, repayment amounts, loan calculators, or terms like \"loan amount\", \"apply now\", interest rates, borrowing durations, or financial incentives.
For each image:
- A risk level (No, Low, Medium, High).
- The specific policy it likely violates.
- A brief explanation of why.
- A suggestion on how to fix it (if applicable).

If the risk_level is *High*, *Medium* or *Low* return Yes; otherwise, return No. Do not provide any explanation.
"""

def extract_frames(video_path, interval_seconds=2):
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Không thể mở video: {video_path}")
        return frames
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frames.append(frame)
            extracted_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Tổng số frames trích xuất: {len(frames)}")
    return frames

def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def check_nsfw_frame(frame, frame_index, api_url):
    try:
        if isinstance(frame, str):
            frame = cv2.imread(frame)
            if frame is None:
                print(f"Frame {frame_index}: Không thể đọc ảnh")
                return (frame_index, "Error")
        
        base64_image = frame_to_base64(frame)
        
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT_TEMPLATE},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1500
        }
        
        response = requests.post(api_url, headers=HEADERS, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            print(f"Frame {frame_index}: {answer}")
            return (frame_index, answer)
        else:
            print(f"Frame {frame_index}: Lỗi API - Status {response.status_code}")
            return (frame_index, "Error")
            
    except Exception as e:
        print(f"Frame {frame_index}: Lỗi - {str(e)}")
        return (frame_index, "Error")

def check_image_nsfw(image_path, api_url):
    frame = cv2.imread(image_path)
    if frame is None:
        print("Không thể đọc ảnh!")
        return "Error"
    
    _, result = check_nsfw_frame(frame, "Image", api_url)
    
    print(f"KẾT QUẢ: {result}")
    
    return result

def check_video_nsfw(video_path, api_url, interval_seconds=2, max_workers=10, threshold_percent=20):
    """
    Kiểm tra video NSFW với logic: cần >= threshold_percent% frames có kết quả "Yes" mới kết luận là "Yes"
    
    Args:
        video_path: Đường dẫn video
        api_url: URL của API
        interval_seconds: Khoảng thời gian giữa các frames
        max_workers: Số threads tối đa
        threshold_percent: Ngưỡng phần trăm (mặc định 30%)
    
    Returns:
        "Yes" nếu >= threshold_percent% frames có "Yes", "No" nếu không
    """
    frames = extract_frames(video_path, interval_seconds)
    
    if not frames:
        print("Không có frame nào được trích xuất!")
        return "No"
    
    print(f"\nBắt đầu kiểm tra {len(frames)} frames với {max_workers} threads...")
    print(f"Ngưỡng: {threshold_percent}% frames phải có kết quả 'Yes' để kết luận vi phạm")
    
    # Thu thập tất cả kết quả
    results = {}
    yes_count = 0
    valid_count = 0  # Số frames hợp lệ (không phải Error)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(check_nsfw_frame, frame, i, api_url): i 
            for i, frame in enumerate(frames)
        }
        
        # Chờ tất cả frames xong (không cancel sớm)
        for future in as_completed(futures):
            frame_index, result = future.result()
            results[frame_index] = result
            
            # Đếm số frames có "Yes" và số frames hợp lệ
            if result.lower().startswith('yes'):
                yes_count += 1
                valid_count += 1
                print(f"Frame {frame_index}: {result} ✓")
            elif result.lower().startswith('no'):
                valid_count += 1
            # Error không tính vào valid_count
    
    # Tính tỷ lệ
    if valid_count == 0:
        print("Không có frame hợp lệ nào!")
        final_result = "No"
    else:
        percentage = (yes_count / valid_count) * 100
        print(f"\n{'='*60}")
        print(f"THỐNG KÊ KẾT QUẢ:")
        print(f"  - Tổng số frames: {len(frames)}")
        print(f"  - Frames hợp lệ: {valid_count}")
        print(f"  - Frames có 'Yes': {yes_count}")
        print(f"  - Tỷ lệ: {percentage:.2f}%")
        print(f"  - Ngưỡng yêu cầu: {threshold_percent}%")
        print(f"{'='*60}")
        
        if percentage >= threshold_percent:
            final_result = "Yes"
            print(f"⚠️  KẾT LUẬN: VI PHẠM (≥{threshold_percent}% frames có 'Yes')")
        else:
            final_result = "No"
            print(f"✅ KẾT LUẬN: AN TOÀN (<{threshold_percent}% frames có 'Yes')")
    
    print(f"\nKẾT QUẢ CUỐI CÙNG: {final_result}")
    
    return final_result

def is_image_file(file_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}
    return Path(file_path).suffix.lower() in image_extensions

def is_video_file(file_path):
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
    return Path(file_path).suffix.lower() in video_extensions

def check_media_nsfw(media_path, api_url, interval_seconds=2, max_workers=10):
    if not os.path.exists(media_path):
        print(f"Lỗi: File không tồn tại - {media_path}")
        return "Error"
    
    if is_image_file(media_path):
        return check_image_nsfw(media_path, api_url)
    elif is_video_file(media_path):
        return check_video_nsfw(media_path, api_url, interval_seconds, max_workers)
    else:
        print(f"Lỗi: File không được hỗ trợ - {media_path}")
        print("Chỉ hỗ trợ: Ảnh (jpg, png, ...) và Video (mp4, avi, ...)")
        return "Error"

# ===========================
# NEW FUNCTIONS - FOLDER SUPPORT
# ===========================

def get_all_media_files(path: str) -> List[str]:
    """
    Lấy tất cả file ảnh và video từ một đường dẫn.
    Nếu path là file → trả về [file]
    Nếu path là folder → scan tất cả file media bên trong
    """
    path_obj = Path(path)
    
    if path_obj.is_file():
        return [str(path_obj)]
    
    elif path_obj.is_dir():
        media_files = []
        for file_path in path_obj.rglob('*'):
            if file_path.is_file() and (is_image_file(str(file_path)) or is_video_file(str(file_path))):
                media_files.append(str(file_path))
        return sorted(media_files)
    
    else:
        print(f"Lỗi: Đường dẫn không tồn tại - {path}")
        return []

def check_multiple_media(media_paths: List[str], api_url: str, interval_seconds=2, max_workers=10) -> Dict[str, str]:
    """
    Kiểm tra nhiều file media và trả về kết quả dạng dictionary
    """
    results = {}
    total = len(media_paths)
    
    print(f"\n{'='*60}")
    print(f"BẮT ĐẦU KIỂM TRA {total} FILE MEDIA")
    print(f"{'='*60}\n")
    
    for idx, media_path in enumerate(media_paths, 1):
        print(f"\n[{idx}/{total}] Đang kiểm tra: {media_path}")
        print("-" * 60)
        
        result = check_media_nsfw(media_path, api_url, interval_seconds, max_workers)
        results[media_path] = result
        
        print(f"Kết quả: {result}")
    
    return results

def print_summary(results: Dict[str, str]):
    """
    In tổng kết kết quả kiểm tra
    """
    print(f"\n{'='*60}")
    print("TỔNG KẾT KẾT QUẢ")
    print(f"{'='*60}\n")
    
    violated = []
    safe = []
    errors = []
    
    for path, result in results.items():
        if result.lower().startswith('yes'):
            violated.append(path)
        elif result.lower().startswith('no'):
            safe.append(path)
        else:
            errors.append(path)
    
    print(f"📊 Tổng số file: {len(results)}")
    print(f"✅ An toàn: {len(safe)}")
    print(f"⚠️  Vi phạm: {len(violated)}")
    print(f"❌ Lỗi: {len(errors)}")
    
    if violated:
        print(f"\n⚠️  DANH SÁCH FILE VI PHẠM:")
        for path in violated:
            print(f"   - {path}")
    
    if errors:
        print(f"\n❌ DANH SÁCH FILE LỖI:")
        for path in errors:
            print(f"   - {path}")
    
    print(f"\n{'='*60}\n")
    
    return len(violated) > 0

if __name__ == "__main__":
    # Cấu hình mặc định
    INTERVAL_SECONDS = 1
    MAX_THREADS = 50
    

    
    INPUT_PATH = "/home/hiepnd72/Documents/work/blocked/imgs/images"
    
    # Lấy tất cả file media
    media_files = get_all_media_files(INPUT_PATH)
    
    if not media_files:
        print("Không tìm thấy file media nào!")
        sys.exit(1)
    
    # Kiểm tra tất cả file
    results = check_multiple_media(media_files, API_URL, INTERVAL_SECONDS, MAX_THREADS)
    
    # In tổng kết
    has_violation = print_summary(results)
    
    # Exit code: 0 nếu tất cả pass, 1 nếu có vi phạm
    sys.exit(1 if has_violation else 0)