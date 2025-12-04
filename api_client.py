import requests
import base64
import cv2
import os
from typing import Tuple, Optional
from config import (
    VLM_API_URL, VLM_MODEL_NAME, VLM_HEADERS,
    TRANSCRIBE_API_URL,
    IMAGE_PROMPT_TEMPLATE, TEXT_PROMPT_TEMPLATE
)


def frame_to_base64(frame) -> str:
    """Chuyển frame (numpy array) thành base64 string"""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')


def transcribe_audio(audio_path: str, api_url: str = TRANSCRIBE_API_URL) -> str:
    """
    Gửi audio file đến API transcribe để lấy text.
    
    Args:
        audio_path: Đường dẫn đến file audio
        api_url: URL của API transcribe
    
    Returns:
        Text transcript từ audio
    """
    try:
        with open(audio_path, 'rb') as audio_file:
            files = {'file': (os.path.basename(audio_path), audio_file, 'audio/wav')}
            
            response = requests.post(api_url, files=files, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                # Giả sử API trả về dạng {"text": "..."} hoặc {"transcript": "..."}
                transcript = result.get('text') or result.get('transcript') or result.get('result', '')
                print(f"Transcribe thành công. Text length: {len(transcript)} characters")
                return transcript
            else:
                print(f"Lỗi API transcribe - Status {response.status_code}: {response.text}")
                return ""
                
    except FileNotFoundError:
        print(f"Không tìm thấy file audio: {audio_path}")
        return ""
    except Exception as e:
        print(f"Lỗi khi transcribe audio: {str(e)}")
        return ""


def check_text_vlm(text: str, api_url: str = VLM_API_URL) -> str:
    """
    Gửi text đến VLM API để kiểm tra vi phạm.
    
    Args:
        text: Text cần kiểm tra
        api_url: URL của VLM API
    
    Returns:
        "Yes" hoặc "No" hoặc "Error"
    """
    if not text or not text.strip():
        print("Text rỗng, trả về No")
        return "No"
    
    try:
        prompt = TEXT_PROMPT_TEMPLATE.format(transcript=text)
        
        payload = {
            "model": VLM_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1500
        }
        
        response = requests.post(api_url, headers=VLM_HEADERS, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            print(f"Text check result: {answer}")
            return answer
        else:
            print(f"Lỗi VLM API (text) - Status {response.status_code}: {response.text}")
            return "Error"
            
    except Exception as e:
        print(f"Lỗi khi kiểm tra text: {str(e)}")
        return "Error"


def check_frame_vlm(frame, frame_index: int, api_url: str = VLM_API_URL) -> Tuple[int, str]:
    """
    Gửi frame đến VLM API để kiểm tra vi phạm.
    
    Args:
        frame: Frame (numpy array) hoặc đường dẫn ảnh
        frame_index: Index của frame
        api_url: URL của VLM API
    
    Returns:
        Tuple (frame_index, result) với result là "Yes", "No", hoặc "Error"
    """
    try:
        # Nếu frame là string (đường dẫn), đọc ảnh
        if isinstance(frame, str):
            frame = cv2.imread(frame)
            if frame is None:
                print(f"Frame {frame_index}: Không thể đọc ảnh")
                return (frame_index, "Error")
        
        base64_image = frame_to_base64(frame)
        
        payload = {
            "model": VLM_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": IMAGE_PROMPT_TEMPLATE},
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
        
        response = requests.post(api_url, headers=VLM_HEADERS, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            print(f"Frame {frame_index}: {answer}")
            return (frame_index, answer)
        else:
            print(f"Frame {frame_index}: Lỗi VLM API - Status {response.status_code}")
            return (frame_index, "Error")
            
    except Exception as e:
        print(f"Frame {frame_index}: Lỗi - {str(e)}")
        return (frame_index, "Error")

