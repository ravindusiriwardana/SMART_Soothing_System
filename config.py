import os

# Audio Settings
SAMPLE_RATE = 16000
SEGMENT_DURATION = 5   
SEGMENT_SIZE = SAMPLE_RATE * SEGMENT_DURATION
MAX_LEN = 157
N_MFCC = 40

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CRY_MODEL_PATH = os.path.join(BASE_DIR, "cry_model", "baby_cry_lstm_complete_acc_64%.h5")
PARENT_VOICE_PATH = os.path.join(BASE_DIR, "audio", "parents_audio", "parent_voice_16k.wav")
MUSIC_BASE_DIR = os.path.join(BASE_DIR, "music", "categorized_music")

# --- RL Q-Table Paths ---
RL_TABLE_PATH = os.path.join(BASE_DIR, "data", "q_table", "q_table.pkl")
MUSIC_RL_TABLE_PATH = os.path.join(BASE_DIR, "data", "q_table", "music_q_table.pkl")

# Server Settings
WS_HOST = "0.0.0.0"
WS_PORT = 8765

# AI Models
TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/your_tts"
LLM_MODEL_NAME = "gemma:2b"

# High-Level Categories (States)
CATEGORIES = [
    'belly pain', 'burping', 'discomfort', 'hungry', 'laugh',
    'lonely', 'noise', 'scared', 'silence', 'tired'
]

# Low-Level Music Categories (Actions for the Music Agent)
# These must match your folder names exactly
MUSIC_CATEGORIES = [
    "loud_high_pitched", 
    "rhythmic_rising_pitch", 
    "medium_pitch_whiny",
    "playful_upbeat", 
    "warm_soft_comforting", 
    "noise",
    "trembling_pitch", 
    "silence"
]