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
    """Handles text generation using local LLM with a robust static fallback."""
    def __init__(self, model_name, parent_name):
        self.model_name = model_name
        self.parent_name = parent_name
        self.llm = None
        self._init_fallbacks()
        self._init_llm()

    def _init_fallbacks(self):
        """Predefined ~100-word soothing sentences for when the LLM is unavailable."""
        self.fallback_phrases = {
            "hungry": (
                "Oh, my hungry little one, I know that tummy is feeling so empty right now. "
                "Shhh, Mommy hears you, and Mommy is getting everything ready for you. "
                "You do not have to cry, my sweet baby, because your milk is coming so very soon. "
                "Let's take a deep breath together while we wait just a tiny moment. Shhh, shhh. "
                "I am right here with you, holding you so tight. You are entirely safe, you are so deeply loved, "
                "and your little tummy will be full and happy in just a minute. Shhh, it is okay."
            ),
            "tired": (
                "Shhh, my sweet baby, it is time to close those heavy eyes. Mommy is right here, "
                "holding you so close and keeping you so warm. The day is done, and everything is quiet and still. "
                "You can let go now and drift away into the softest, most beautiful dreams. "
                "Shhh, shhh, breathe with me, nice and slow. I am watching over you, and you are entirely safe in my arms. "
                "Sleep now, my little love, sleep and rest your little body. Mommy loves you so much. Shhh... just sleep."
            ),
            "belly pain": (
                "Oh, my darling, I know that tiny tummy is feeling so tight and uncomfortable right now. "
                "Shhh, Mommy is here to rub it and make the pressure go away. Let's just rock together, nice and slow. "
                "Shhh, shhh, breathe with me. You are so brave, and this uncomfortable feeling will pass very soon. "
                "I am holding you so close, keeping you warm and safe. Just rest against my chest, listen to my heartbeat, "
                "and let your little body relax. Shhh, everything is going to be alright, my sweet love."
            ),
            "scared": (
                "Shhh, my precious baby, Mommy is right here. You are not alone, and there is absolutely nothing to be afraid of. "
                "I am holding you so securely in my arms, where nothing can ever hurt you. Shhh, shhh, just listen to the sound of my voice. "
                "Feel how warm and cozy it is right here. Whatever startled you is gone now, and only love is left. "
                "You are my beautiful baby, and I will always be right here whenever you need me. Rest now, feel safe, and just breathe."
            ),
            "lonely": (
                "Shhh, I am right here, my beautiful baby. You are never alone because Mommy is always watching over you. "
                "I hear you calling for me, and I came as fast as I could to hold you close. "
                "Feel how warm my arms are? Shhh, shhh, let's just snuggle together. You are the most important thing in the world to me. "
                "Just listen to my voice, nice and slow. I am here, I am staying right here, and you are so incredibly loved. Shhh, rest now."
            ),
            "default": (
                "Shhh, my sweet little one, Mommy is right here with you. I know you are feeling upset right now, "
                "but you are completely safe. Let's just take a slow, deep breath together. Shhh, shhh. "
                "I am holding you so close, and I am not going anywhere. Everything is quiet, everything is calm, "
                "and whatever is bothering you will pass so very soon. Just listen to the rhythm of my voice and let your little body relax. "
                "Mommy loves you more than anything. Shhh, rest now, everything is perfectly okay."
            )
        }

    def _init_llm(self):
        if ChatOllama:
            try:
                print(f"🔄 Connecting to LLM ({self.model_name})...")
                self.llm = ChatOllama(model=self.model_name, temperature=0.5)
                
                system_instruction = (
                    f"You are a loving, deeply patient parent named {self.parent_name}. "
                    "Your infant is distressed and needs to hear your gentle voice to feel safe. "
                    "I will provide the reason for their crying in the user message. "
                    "Your task: Generate a soothing, sensitive monologue of exactly 60 to 70 words. "
                    "Guidelines: "
                    "- Use 'Parentese': slow, rhythmic, and repetitive language. "
                    "- Use soft, comforting sounds (shhh, ooh, mmm). "
                    "- Acknowledge their feeling gently but quickly pivot to peace. "
                    "- Focus on sensory comfort: warmth, safety, and being held. "
                    "- Do NOT use emojis, hashtags, or bracketed actions."
                )
                
                self.prompt = ChatPromptTemplate.from_messages([
                    ("system", system_instruction),
                    ("human", "{human}"),
                ])
                print(f"✅ LLM Service ready: {self.model_name}")
            except Exception as e:
                print(f"❌ LLM Init Failed: {e}. Falling back to predefined phrases.")
        else:
            print("⚠️ 'langchain_ollama' not installed. Using fallback text.")
    
    def generate_phrase(self, emotion):
        """Attempts to generate via LLM, but instantly falls back to predefined phrases on failure."""
        if self.llm:
            try:
                chain = self.prompt | self.llm
                response = chain.invoke({"human": f"The baby is feeling: {emotion}"})
                return response.content
            except Exception as e:
                print(f"⚠️ LLM Generation Error during runtime: {e}. Using fallback.")
        
        # If LLM is not initialized or threw an error, use the fallback dictionary
        return self.fallback_phrases.get(emotion, self.fallback_phrases["default"])


class TTSService:
    """Handles Text-to-Speech synthesis and Audio Processing."""
    def __init__(self, model_name, device="cpu"):
        self.model_name = model_name
        self.synthesizer = None
        self._init_tts(device)

    def _init_tts(self, device):
        try:
            print(f"🔄 Loading TTS Model ({self.model_name}). This may take a while...")
            # gpu=False is usually required for Mac M1/M2 and Raspberry Pi for this specific library
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
        """Strips out special characters that might break the TTS engine."""
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
        
        args = {
            "text": cleaned_text, 
            "file_path": output_file,
            "language": "en"  # Crucial parameter for YourTTS
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
        except Exception as e:
            print(f"⚠️ Playback error: {e}")



#OLD Code
# import os
# import re
# import unicodedata
# import numpy as np
# import librosa
# import soundfile as sf
# import pygame
# import time
# from TTS.api import TTS

# # Optional Imports for LLM
# try:
#     from langchain_ollama import ChatOllama
#     from langchain_core.prompts import ChatPromptTemplate
# except ImportError:
#     ChatOllama = None

# class LLMService:
#     """Handles text generation using local LLM."""
#     def __init__(self, model_name, parent_name):
#         self.model_name = model_name
#         self.parent_name = parent_name
#         self.llm = None
#         self._init_llm()

#     def _init_llm(self):
#         if ChatOllama:
#             try:
#                 print(f"🔄 Connecting to LLM ({self.model_name})...")
#                 self.llm = ChatOllama(model=self.model_name, temperature=0.7)
                
#                 # Setup prompt
#                 # system_instruction = (
#                 #     f"You are a loving parent named {self.parent_name}. "
#                 #     "Your infant is crying. I will tell you the reason. "
#                 #     "Generate a short, soothing phrase (max 15 words) tailored to that reason. "
#                 #     "Do not use emojis or actions like *hugs*."
#                 # )
#                 system_instruction = (
#                     f"You are a loving, deeply patient parent named {self.parent_name}. "
#                     "Your infant is distressed and needs to hear your gentle voice to feel safe. "
#                     "I will provide the reason for their crying. "
#                     "Your task: Generate a soothing, sensitive monologue of 60 to 70 words. "
#                     "Guidelines: "
#                     "- Use 'Parentese': slow, rhythmic, and repetitive language. "
#                     "- Use soft, comforting sounds (shhh, ooh, mmm). "
#                     "- Acknowledge their feeling (e.g., 'I know your tummy hurts') but quickly pivot to peace. "
#                     "- Focus on sensory comfort: warmth, safety, and being held. "
#                     "- Do NOT use emojis, hashtags, or bracketed actions like *kisses*."
#                 )

#                 system_instruction = (
#                     f"You are a loving parent named {self.parent_name}. Your infant is crying. "
#                     "Your goal is to provide a 30-second rhythmic, soothing monologue. "
#                     "\n\nSTRICT GUIDELINES:"
#                     "\n1. LENGTH: Write exactly 60 to 75 words. This is critical for the 30-second timing."
#                     "\n2. TONE: Use 'Parentese'—warm, gentle, and highly repetitive. Focus on long vowel sounds (e.g., 'sleepy', 'cozy', 'safe')."
#                     "\n3. STRUCTURE: Use short, simple sentences. Start and end the monologue with 'Shhh... shhh...'"
#                     f"\n4. TAILORING: Subtly address the reason for crying ({emotion}) without being clinical. "
#                     "If hungry, mention 'warm milk' or 'filling the tummy'. If tired, mention 'heavy eyes' and 'soft blankets'."
#                     "\n5. NO EMOJIS: Do not use any emojis, asterisks, or stage directions like *hugs*."
#                     "\n6. PACING: Use commas and periods frequently to create natural pauses for the TTS model."
#                 )
                
#                 self.prompt = ChatPromptTemplate.from_messages([
#                     ("system", system_instruction),
#                     ("human", "{human}"),
#                 ])
#                 print(f"✅ LLM Service ready: {self.model_name}")
#             except Exception as e:
#                 print(f"❌ LLM Init Failed: {e}")
#         else:
#             print("⚠️ 'langchain_ollama' not installed. Using fallback text.")
    
#     def generate_phrase(self, emotion):
#         if not self.llm:
#             return "Shhh, mommy is here. Everything is okay."
#         try:
#             chain = self.prompt | self.llm
#             response = chain.invoke({"human": f"The baby is feeling: {emotion}"})
#             return response.content
#         except Exception as e:
#             print(f"⚠️ LLM Generation Error: {e}")
#             return "Shhh, it's okay, I am here."

# class TTSService:
#     """Handles Text-to-Speech synthesis and Audio Processing."""
#     def __init__(self, model_name, device="cpu"):
#         self.model_name = model_name
#         self.synthesizer = None
#         self._init_tts(device)

#     def _init_tts(self, device):
#         try:
#             print(f"🔄 Loading TTS Model ({self.model_name}). This may take a while...")
#             # gpu=True if you have CUDA, else False (Mac M1/M2 usually requires gpu=False for this lib)
#             self.synthesizer = TTS(model_name=self.model_name, gpu=False) 
#             print(f"✅ TTS Service ready")
#         except Exception as e:
#             print(f"❌ TTS Init Failed: {e}")

#     def preprocess_voice(self, input_path, output_path, sr=16000):
#         """Cleans and normalizes the parent's voice sample."""
#         try:
#             y, sr = librosa.load(input_path, sr=sr)
#             y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            
#             # Spectral gating (denoising)
#             stft = librosa.stft(y_trimmed)
#             magnitude, phase = librosa.magphase(stft)
#             noise_est = np.median(magnitude, axis=1, keepdims=True)
#             mask = magnitude >= noise_est
#             stft_clean = stft * mask
#             y_denoised = librosa.istft(stft_clean)
            
#             y_normalized = librosa.util.normalize(y_denoised)
#             sf.write(output_path, y_normalized, sr)
#             return output_path
#         except Exception as e:
#             print(f"⚠️ Voice processing error: {e}")
#             return None

#     def clean_text(self, text):
#         if not text: return ""
#         text = text.encode("ascii", "ignore").decode()
#         text = unicodedata.normalize("NFKD", text)
#         text = "".join(c for c in text if not unicodedata.combining(c))
#         text = re.sub(r"[^a-zA-Z0-9.,!?'\- ]+", " ", text)
#         return re.sub(r"\s+", " ", text).strip()

#     def synthesize_and_play(self, text, output_file, speaker_wav=None):
#         if not self.synthesizer:
#             print("⚠️ TTS not available.")
#             return

#         cleaned_text = self.clean_text(text)
#         print(f"🗣️ Speaking: '{cleaned_text}'")
        
#         # --- FIX: Changed "language_name" to "language" ---
#         args = {
#             "text": cleaned_text, 
#             "file_path": output_file,
#             "language": "en"  # <--- This keyword is critical
#         }
        
#         if self.synthesizer.is_multi_speaker and speaker_wav:
#             args["speaker_wav"] = speaker_wav

#         try:
#             self.synthesizer.tts_to_file(**args)
#             self._play_audio(output_file)
#         except Exception as e:
#             print(f"❌ Synthesis error: {e}")

#     def _play_audio(self, file_path):
#         try:
#             pygame.mixer.init()
#             pygame.mixer.music.load(file_path)
#             pygame.mixer.music.play()
#             while pygame.mixer.music.get_busy():
#                 time.sleep(0.1)
#             # Keeping mixer alive to prevent lag on next play
#         except Exception as e:
#             print(f"⚠️ Playback error: {e}")