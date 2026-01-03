# config.py
import os

# Audio Settings
SAMPLE_RATE = 16000
SEGMENT_DURATION = 10  # seconds
SEGMENT_SIZE = SAMPLE_RATE * SEGMENT_DURATION
N_MFCC = 40
MAX_LEN = 216

# AI / Model Paths (Update these if your files are in different spots)
# Using absolute paths or relative to the main project folder is safer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CRY_MODEL_PATH = os.path.join(BASE_DIR, "cry_model", "cry_lstm_model00.h5")
RL_TABLE_PATH = os.path.join(BASE_DIR, "data", "q_table", "q_table.pkl")
PARENT_VOICE_PATH = os.path.join(BASE_DIR, "audio", "parents_audio", "parent_voice_16k.wav")
MUSIC_BASE_DIR = os.path.join(BASE_DIR, "music", "categorized_music")

# Model Names
TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/your_tts"
LLM_MODEL_NAME = "gemma:2b"

# Server Settings
WS_HOST = "0.0.0.0"
WS_PORT = 8765

# Categories
CATEGORIES = [
    'belly pain', 'burping', 'discomfort', 'hungry', 'laugh',
    'lonely', 'noise', 'scared', 'silence', 'tired'
]