import os
import random
import time
import pygame
from config import MUSIC_BASE_DIR

class MusicPlayer:
    def __init__(self):
        # Initialize the audio mixer
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        
        self.base_dir = MUSIC_BASE_DIR
        
        # --- UPDATED MAPPING TO MATCH YOUR FOLDERS EXACTLY ---
        self.emotion_map = {
            "belly pain": "loud_high_pitched",
            "burping":    "rhythmic_rising_pitch",
            "discomfort": "medium_pitch_whiny",
            "hungry":     "rhythmic_rising_pitch",
            "laugh":      "playful_upbeat",
            "lonely":     "warm_soft_comforting",
            "noise":      "noise",                 # CHANGED: "neutral_noise_blocker" -> "noise"
            "scared":     "trembling_pitch",
            "silence":    "silence",               # CHANGED: "baseline_state" -> "silence"
            "tired":      "warm_soft_comforting"   # CHANGED: "low_energy_whining" -> "warm_soft_comforting" (Fallback since folder is missing)
        }

    def _get_category_path(self, emotion):
        """Helper to resolve the folder path for a given emotion."""
        # Default to 'baseline_state' if the emotion is unknown
        category = self.emotion_map.get(emotion.lower(), "baseline_state")
        return os.path.join(self.base_dir, category)

    def play_music(self, emotion):
        """Plays a random song associated with the detected emotion."""
        print(f"🎵 Music Player received emotion: {emotion}")
        
        folder_path = self._get_category_path(emotion)
        
        if not os.path.exists(folder_path):
            print(f"⚠️ Music category folder not found: {folder_path}")
            return

        # List valid audio files
        try:
            songs = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp3', '.wav'))]
        except Exception as e:
            print(f"⚠️ Error accessing music folder: {e}")
            return

        if not songs:
            print(f"⚠️ No songs found in {folder_path}")
            return

        # Pick a random song
        song_file = random.choice(songs)
        song_path = os.path.join(folder_path, song_file)
        
        try:
            print(f"▶️ Playing: {song_file}")
            pygame.mixer.music.load(song_path)
            pygame.mixer.music.play()
            
            # Non-blocking check so the loop doesn't freeze entirely, 
            # but we wait a bit to prevent rapid-fire skipping
            while pygame.mixer.music.get_busy():
                time.sleep(1)
                
        except Exception as e:
            print(f"❌ Error playing music: {e}")

    def stop(self):
        """Stops any currently playing music."""
        pygame.mixer.music.stop()