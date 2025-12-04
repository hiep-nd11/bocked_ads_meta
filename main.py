#!/usr/bin/env python3
"""
Main script để kiểm tra video theo Meta Advertising Policy.

Quy trình:
1. Nhận video làm đầu vào
2. Tách audio từ video
3. Gửi audio đến API transcribe để lấy text
4. Gửi text qua VLM để kiểm tra vi phạm
5. Trích xuất frames từ video
6. Gửi frames qua VLM để kiểm tra vi phạm
7. Tổng hợp: Nếu 1 trong 2 (text hoặc frames) có Yes thì kết luận là Yes
"""

import sys
import os
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from video_utils import extract_frames, extract_audio, is_video_file
from api_client import transcribe_audio, check_text_vlm, check_frame_vlm
from config import DEFAULT_INTERVAL_SECONDS, DEFAULT_MAX_THREADS


def check_video_frames(frames, max_workers: int = 50) -> str:
    """
    Kiểm tra tất cả frames của video.
    
    Args:
        frames: List các frames
        max_workers: Số threads tối đa
    
    Returns:
        "Yes" nếu có vi phạm, "No" nếu không, "Error" nếu có lỗi
    """
    if not frames:
        print("Không có frame nào được trích xuất!")
        return "No"
    
    print(f"\nBắt đầu kiểm tra {len(frames)} frames với {max_workers} threads...")
    
    final_result = "No"
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(check_frame_vlm, frame, i): i 
            for i, frame in enumerate(frames)
        }
        
        for future in as_completed(futures):
            frame_index, result = future.result()
            
            if result.lower().startswith('yes'):
                print(f"\n⚠️  Phát hiện vi phạm tại frame {frame_index}!")
                final_result = "Yes"
                # Cancel các tasks còn lại
                for f in futures:
                    f.cancel()
                break
    
    print(f"KẾT QUẢ KIỂM TRA FRAMES: {final_result}")
    return final_result


def check_video_complete(video_path: str, 
                        interval_seconds: float = 1,
                        max_workers: int = 50,
                        keep_audio: bool = False) -> str:
    """
    Kiểm tra video đầy đủ: cả audio (text) và frames.
    
    Args:
        video_path: Đường dẫn đến file video
        interval_seconds: Khoảng thời gian giữa các frames (giây)
        max_workers: Số threads tối đa cho việc kiểm tra frames
        keep_audio: Có giữ lại file audio sau khi xử lý không
    
    Returns:
        "Yes" nếu có vi phạm (từ text hoặc frames), "No" nếu không, "Error" nếu có lỗi
    """
    if not os.path.exists(video_path):
        print(f"Lỗi: File không tồn tại - {video_path}")
        return "Error"
    
    if not is_video_file(video_path):
        print(f"Lỗi: File không phải là video - {video_path}")
        return "Error"
    
    print(f"\n{'='*60}")
    print(f"BẮT ĐẦU KIỂM TRA VIDEO: {video_path}")
    print(f"{'='*60}\n")
    
    # ==========================================
    # BƯỚC 1: Tách audio từ video
    # ==========================================
    print("📢 BƯỚC 1: Tách audio từ video...")
    try:
        audio_path = extract_audio(video_path)
        print(f"✅ Audio đã được tách: {audio_path}\n")
    except Exception as e:
        print(f"❌ Lỗi khi tách audio: {str(e)}")
        audio_path = None
    
    # ==========================================
    # BƯỚC 2: Transcribe audio → text
    # ==========================================
    text_result = "No"
    transcript = ""
    
    if audio_path and os.path.exists(audio_path):
        print("🎤 BƯỚC 2: Transcribe audio thành text...")
        transcript = transcribe_audio(audio_path)
        
        if transcript:
            print(f"✅ Transcribe thành công\n")
            
            # ==========================================
            # BƯỚC 3: Kiểm tra text qua VLM
            # ==========================================
            print("📝 BƯỚC 3: Kiểm tra text qua VLM...")
            text_result = check_text_vlm(transcript)
            print(f"KẾT QUẢ KIỂM TRA TEXT: {text_result}\n")
        else:
            print("⚠️  Không có transcript, bỏ qua kiểm tra text\n")
        
        # Xóa file audio nếu không cần giữ lại
        if not keep_audio:
            try:
                os.remove(audio_path)
                print(f"🗑️  Đã xóa file audio tạm: {audio_path}\n")
            except:
                pass
    else:
        print("⚠️  Không có audio, bỏ qua kiểm tra text\n")
    
    # ==========================================
    # BƯỚC 4: Trích xuất frames từ video
    # ==========================================
    print("🖼️  BƯỚC 4: Trích xuất frames từ video...")
    frames = extract_frames(video_path, interval_seconds)
    
    # ==========================================
    # BƯỚC 5: Kiểm tra frames qua VLM
    # ==========================================
    frames_result = "No"
    if frames:
        print(f"✅ Đã trích xuất {len(frames)} frames\n")
        print("🔍 BƯỚC 5: Kiểm tra frames qua VLM...")
        frames_result = check_video_frames(frames, max_workers)
    else:
        print("⚠️  Không có frames để kiểm tra\n")
    
    # ==========================================
    # BƯỚC 6: Tổng hợp kết quả
    # ==========================================
    print(f"\n{'='*60}")
    print("TỔNG HỢP KẾT QUẢ")
    print(f"{'='*60}\n")
    
    print(f"📝 Kết quả kiểm tra TEXT: {text_result}")
    print(f"🖼️  Kết quả kiểm tra FRAMES: {frames_result}\n")
    
    # Nếu 1 trong 2 có Yes thì kết luận là Yes
    final_result = "Yes" if (
        text_result.lower().startswith('yes') or 
        frames_result.lower().startswith('yes')
    ) else "No"
    
    print(f"🎯 KẾT QUẢ CUỐI CÙNG: {final_result}")
    print(f"{'='*60}\n")
    
    return final_result


def main():
    parser = argparse.ArgumentParser(
        description='Kiểm tra video theo Meta Advertising Policy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python main.py video.mp4
  python main.py video.mp4 --interval 2 --threads 30
  python main.py video.mp4 --keep-audio
        """
    )
    
    parser.add_argument(
        '--video_path',
        type=str,
        default="/home/hiepnd72/Documents/work/blocked/12.11/Drama/Drama (5).mp4",
        help='Đường dẫn đến file video cần kiểm tra'
    )
    
    parser.add_argument(
        '--interval',
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help=f'Khoảng thời gian giữa các frames (giây, mặc định: {DEFAULT_INTERVAL_SECONDS})'
    )
    
    parser.add_argument(
        '--threads',
        type=int,
        default=DEFAULT_MAX_THREADS,
        help=f'Số threads tối đa cho việc kiểm tra frames (mặc định: {DEFAULT_MAX_THREADS})'
    )
    
    parser.add_argument(
        '--keep-audio',
        action='store_true',
        help='Giữ lại file audio sau khi xử lý (mặc định: xóa)'
    )
    
    args = parser.parse_args()
    
    # Kiểm tra video path
    if not os.path.exists(args.video_path):
        print(f"❌ Lỗi: File không tồn tại - {args.video_path}")
        sys.exit(1)
    
    if not is_video_file(args.video_path):
        print(f"❌ Lỗi: File không phải là video - {args.video_path}")
        sys.exit(1)
    
    # Kiểm tra video
    result = check_video_complete(
        args.video_path,
        interval_seconds=args.interval,
        max_workers=args.threads,
        keep_audio=args.keep_audio
    )
    
    # Exit code: 0 nếu pass, 1 nếu có vi phạm
    sys.exit(1 if result.lower().startswith('yes') else 0)


if __name__ == "__main__":
    main()

