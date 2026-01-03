import os
import re
import unicodedata
import numpy as np
import librosa
import soundfile as sf
import pygame
import time
from TTS.api import TTS

# Optional Imports for LLM
try:
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    ChatOllama = None

class LLMService:
    """Handles text generation using local LLM."""
    def __init__(self, model_name, parent_name):
        self.model_name = model_name
        self.parent_name = parent_name
        self.llm = None
        self._init_llm()

    def _init_llm(self):
        if ChatOllama:
            try:
                print(f"🔄 Connecting to LLM ({self.model_name})...")
                self.llm = ChatOllama(model=self.model_name, temperature=0.7)
                
                # Setup prompt
                system_instruction = (
                    f"You are a loving parent named {self.parent_name}. "
                    "Your infant is crying. I will tell you the reason. "
                    "Generate a short, soothing phrase (max 15 words) tailored to that reason. "
                    "Do not use emojis or actions like *hugs*."
                )
                
                self.prompt = ChatPromptTemplate.from_messages([
                    ("system", system_instruction),
                    ("human", "{human}"),
                ])
                print(f"✅ LLM Service ready: {self.model_name}")
            except Exception as e:
                print(f"❌ LLM Init Failed: {e}")
        else:
            print("⚠️ 'langchain_ollama' not installed. Using fallback text.")
    
    def generate_phrase(self, emotion):
        if not self.llm:
            return "Shhh, mommy is here. Everything is okay."
        try:
            chain = self.prompt | self.llm
            response = chain.invoke({"human": f"The baby is feeling: {emotion}"})
            return response.content
        except Exception as e:
            print(f"⚠️ LLM Generation Error: {e}")
            return "Shhh, it's okay, I am here."

class TTSService:
    """Handles Text-to-Speech synthesis and Audio Processing."""
    def __init__(self, model_name, device="cpu"):
        self.model_name = model_name
        self.synthesizer = None
        self._init_tts(device)

    def _init_tts(self, device):
        try:
            print(f"🔄 Loading TTS Model ({self.model_name}). This may take a while...")
            # gpu=True if you have CUDA, else False (Mac M1/M2 usually requires gpu=False for this lib)
            self.synthesizer = TTS(model_name=self.model_name, gpu=False) 
            print(f"✅ TTS Service ready")
        except Exception as e:
            print(f"❌ TTS Init Failed: {e}")

    def preprocess_voice(self, input_path, output_path, sr=16000):
        """Cleans and normalizes the parent's voice sample."""
        try:
            y, sr = librosa.load(input_path, sr=sr)
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            
            # Spectral gating (denoising)
            stft = librosa.stft(y_trimmed)
            magnitude, phase = librosa.magphase(stft)
            noise_est = np.median(magnitude, axis=1, keepdims=True)
            mask = magnitude >= noise_est
            stft_clean = stft * mask
            y_denoised = librosa.istft(stft_clean)
            
            y_normalized = librosa.util.normalize(y_denoised)
            sf.write(output_path, y_normalized, sr)
            return output_path
        except Exception as e:
            print(f"⚠️ Voice processing error: {e}")
            return None

    def clean_text(self, text):
        if not text: return ""
        text = text.encode("ascii", "ignore").decode()
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
        text = re.sub(r"[^a-zA-Z0-9.,!?'\- ]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def synthesize_and_play(self, text, output_file, speaker_wav=None):
        if not self.synthesizer:
            print("⚠️ TTS not available.")
            return

        cleaned_text = self.clean_text(text)
        print(f"🗣️ Speaking: '{cleaned_text}'")
        
        # --- FIX: Changed "language_name" to "language" ---
        args = {
            "text": cleaned_text, 
            "file_path": output_file,
            "language": "en"  # <--- This keyword is critical
        }
        
        if self.synthesizer.is_multi_speaker and speaker_wav:
            args["speaker_wav"] = speaker_wav

        try:
            self.synthesizer.tts_to_file(**args)
            self._play_audio(output_file)
        except Exception as e:
            print(f"❌ Synthesis error: {e}")

    def _play_audio(self, file_path):
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            # Keeping mixer alive to prevent lag on next play
        except Exception as e:
            print(f"⚠️ Playback error: {e}")