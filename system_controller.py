import time
import random
import sounddevice as sd
from config import *

from audio.audio_utils import AudioBuffer
from cry_model.cry_classifier import CryClassifier
from rl_agent.q_learning_agent import QLearningAgent
from tts_soother.parent_soother import ParentSoother
from music.music_player import MusicPlayer
from websocket_server.server import WebSocketServer

class SmartCradleSystem:
    def __init__(self):
        print("🚀 Initializing Smart Soothing System...")
        
        self.audio_buffer = AudioBuffer(SEGMENT_SIZE)
        
        print("⏳ Starting WebSocket Server...")
        self.ws_server = WebSocketServer(WS_HOST, WS_PORT)
        
        print("⏳ Loading Cry Classifier...")
        self.cry_classifier = CryClassifier(CRY_MODEL_PATH, CATEGORIES)
        
        print("⏳ Loading Main RL Agent (Voice vs Music)...")
        self.agent = QLearningAgent(CATEGORIES, ["voice", "music"])
        self.agent.load(RL_TABLE_PATH)
        
        print("⏳ Initializing Parent Soother...")
        self.soother = ParentSoother(
            llm_model=LLM_MODEL_NAME,
            tts_model=TTS_MODEL_NAME,
            parent_name="Mommy",
            parent_voice_path=PARENT_VOICE_PATH
        )
        
        print("⏳ Initializing Music Player...")
        self.music_player = MusicPlayer()
        
        self.stream = None
        self.running = False

    def _detect_posture(self):
        return random.choice(["safe", "risky"])

    def start_audio_stream(self):
        self.stream = sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            callback=self.audio_buffer.callback,
            blocksize=1024
        )
        self.stream.start()
        print("🎙️ Audio Stream Started")

    def run(self):
        self.ws_server.start()
        self.start_audio_stream()
        self.running = True
        
        print("✅ System Operational. Listening for cries...")
        
        try:
            while self.running:
                time.sleep(1) 
                
                segment = self.audio_buffer.get_audio_segment()
                if len(segment) < SEGMENT_SIZE:
                    continue

                # 1. Predict Initial Emotion (Current State)
                current_emotion, confidence = self.cry_classifier.predict(segment)
                posture = self._detect_posture()
                
                self.ws_server.broadcast_data({
                    "emotion": current_emotion,
                    "confidence": confidence,
                    "posture": posture
                })

                if current_emotion == "silence":
                    print(f"😴 Baby is silent. Checking again in 60s...")
                    time.sleep(60)
                
                else:
                    print(f"🚨 Analysis: {current_emotion} ({confidence:.2f})")
                    
                    # 2. Decide main action (Voice vs Music)
                    action = self.agent.choose_action(current_emotion)
                    print(f"🤖 Main Agent Decided: {action}")

                    chosen_music_category = None
                    if action == "voice":
                        self.soother.soothe(current_emotion)
                    elif action == "music":
                        chosen_music_category = self.music_player.play_music(current_emotion)
                    
                    # 3. Wait and observe the effect
                    print("⏳ Soothing applied. Waiting 10s to observe effect...")
                    time.sleep(10)
                    
                    # 4. Measure the Next State
                    next_segment = self.audio_buffer.get_audio_segment()
                    next_emotion, next_conf = self.cry_classifier.predict(next_segment)
                    
                    # 5. Calculate Reward
                    if next_emotion == "silence" or next_emotion == "laugh":
                        reward = 10  # Highly successful
                    elif next_emotion == current_emotion:
                        reward = -1  # No improvement, still crying the same way
                    else:
                        reward = 0   # Changed cry type, neutral result
                    
                    # 6. Update Agents
                    # Update High-Level Agent
                    self.agent.update(current_emotion, action, reward, next_emotion)
                    self.agent.save(RL_TABLE_PATH)
                    
                    # Update Low-Level Music Agent (if music was used)
                    if action == "music" and chosen_music_category:
                        self.music_player.update_agent(current_emotion, chosen_music_category, reward, next_emotion)
                    
                    print(f"📈 RL Updated | State: {current_emotion} -> Next: {next_emotion} | Reward: {reward}")

        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self):
        print("\n🛑 Shutting down system...")
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        print("👋 Goodbye.")