# ===========================
# CONFIG
# ===========================

# VLM API Config
VLM_API_URL = "http://162.213.119.141:40484/v1/chat/completions"
VLM_MODEL_NAME = "vlm-7b"
VLM_API_KEY = "mysecretkey123"

VLM_HEADERS = {
    "Authorization": f"Bearer {VLM_API_KEY}"
}

# Transcribe API Config
TRANSCRIBE_API_URL = "http://162.213.119.141:40396/transcribe"

# ===========================
# PROMPT TEMPLATES
# ===========================

# Prompt cho phân tích hình ảnh/frames
IMAGE_PROMPT_TEMPLATE = """
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

For each image:
- A risk level (No, Low, Medium, High).
- The specific policy it likely violates.
- A brief explanation of why.
- A suggestion on how to fix it (if applicable).

If the risk_level is *High* or *Medium*, return Yes; otherwise, return No. Do not provide any explanation.
"""


# Prompt cho phân tích text (transcript từ audio)
TEXT_PROMPT_TEMPLATE = """
Act as a strict Meta (Facebook/Instagram) Advertising Policy compliance expert.

Analyze the following text transcript from a video's audio. This text is intended to be used as ad creative content.
Your task is to identify potential policy violations based on Meta's Advertising Standards.

Focus specifically on the following policies:
1. Adult Content & Sexual Suggestiveness: 
   - Check for sexual references, explicit language, or suggestive content.
   - Check for inappropriate sexual innuendos or double entendres.
2. Sensational Content: Are there words or phrases that might be considered shocking, scary, or inappropriate?
3. Prohibited Content: Check for hate speech, violence, illegal activities, or other prohibited content.

For the text:
- A risk level (No, Low, Medium, High).
- The specific policy it likely violates.
- A brief explanation of why.
- A suggestion on how to fix it (if applicable).

If the risk_level is *High*, *Medium* or *Low*, return Yes; otherwise, return No. Do not provide any explanation.

Text to analyze:
{transcript}
"""

# Default settings
DEFAULT_INTERVAL_SECONDS = 1
DEFAULT_MAX_THREADS = 50
DEFAULT_THRESHOLD_PERCENT = 25  

