import os
import random
import time
import pygame
from config import MUSIC_BASE_DIR, MUSIC_CATEGORIES, MUSIC_RL_TABLE_PATH, CATEGORIES
from rl_agent.q_learning_agent import QLearningAgent

class MusicPlayer:
    def __init__(self):
        # Initialize the audio mixer
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        
        self.base_dir = MUSIC_BASE_DIR
        
        # Initialize the RL Agent specifically for Music Selection
        # States = Baby's Emotions | Actions = Music Folders/Categories
        self.agent = QLearningAgent(states=CATEGORIES, actions=MUSIC_CATEGORIES)
        self.agent.load(MUSIC_RL_TABLE_PATH)

    def play_music(self, emotion):
        """Plays a song based on the RL agent's dynamic selection."""
        print(f"🎵 Music Player received emotion: {emotion}")
        
        # RL Agent decides the music category (Action) based on emotion (State)
        chosen_category = self.agent.choose_action(emotion)
        print(f"🧠 Music RL Agent selected category: {chosen_category}")
        
        folder_path = os.path.join(self.base_dir, chosen_category)
        
        if not os.path.exists(folder_path):
            print(f"⚠️ Music category folder not found: {folder_path}")
            return None

        try:
            songs = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp3', '.wav'))]
        except Exception as e:
            print(f"⚠️ Error accessing music folder: {e}")
            return None

        if not songs:
            print(f"⚠️ No songs found in {folder_path}")
            return None

        # Pick a random song from the chosen category
        song_file = random.choice(songs)
        song_path = os.path.join(folder_path, song_file)
        
        try:
            print(f"▶️ Playing: {song_file}")
            pygame.mixer.music.load(song_path)
            pygame.mixer.music.play()
            
            # Non-blocking wait
            while pygame.mixer.music.get_busy():
                time.sleep(1)
                
        except Exception as e:
            print(f"❌ Error playing music: {e}")
            
        # Return the chosen category so the system_controller can grade its success
        return chosen_category

    def stop(self):
        """Stops any currently playing music."""
        pygame.mixer.music.stop()
        
    def update_agent(self, state, action, reward, next_state):
        """Updates the Q-table for music preferences based on the reward."""
        self.agent.update(state, action, reward, next_state)
        self.agent.save(MUSIC_RL_TABLE_PATH)