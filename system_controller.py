import time
import random
import sounddevice as sd
from config import *

# Component Imports
from audio.audio_utils import AudioBuffer
from cry_model.cry_classifier import CryClassifier
from rl_agent.q_learning_agent import QLearningAgent
from tts_soother.parent_soother import ParentSoother
from music.music_player import MusicPlayer
from websocket_server.server import WebSocketServer

class SmartCradleSystem:
    def __init__(self):
        print("🚀 Initializing Smart Cradle System...")
        
        # 1. Initialize Components
        self.audio_buffer = AudioBuffer(SEGMENT_SIZE)
        
        print("⏳ Starting WebSocket Server...")
        self.ws_server = WebSocketServer(WS_HOST, WS_PORT)
        
        print("⏳ Loading Cry Classifier...")
        self.cry_classifier = CryClassifier(CRY_MODEL_PATH, CATEGORIES)
        
        print("⏳ Loading RL Agent...")
        self.agent = QLearningAgent(CATEGORIES, ["voice", "music"])
        self.agent.load(RL_TABLE_PATH)
        
        print("⏳ Initializing Parent Soother (This includes TTS/LLM loading)...")
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
        """Simulated sensor data."""
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
        """Main execution loop."""
        self.ws_server.start()
        self.start_audio_stream()
        self.running = True
        
        print("✅ System Operational. Listening for cries...")
        
        # Default sleep time (start quick to verify system is working)
        next_sleep_interval = 5 
        
        try:
            while self.running:
                # Sleep dynamically based on the previous state
                time.sleep(next_sleep_interval) 
                
                # Get Audio from the buffer
                segment = self.audio_buffer.get_audio_segment()
                if len(segment) < SEGMENT_SIZE:
                    continue

                # 1. Predict Emotion
                emotion, confidence = self.cry_classifier.predict(segment)
                posture = self._detect_posture()
                
                # 2. Send Data to Flutter App
                self.ws_server.broadcast_data({
                    "emotion": emotion,
                    "confidence": confidence,
                    "posture": posture
                })

                # 3. Logic: Silent vs Crying
                if emotion == "silence":
                    print(f"😴 Baby is silent. Checking again in 60s...")
                    next_sleep_interval = 60  # Long sleep to save resources
                
                else:
                    # Baby is crying!
                    print(f"🚨 Analysis: {emotion} ({confidence:.2f}) | Posture: {posture}")
                    print(f"⚡ Taking action for: {emotion}")
                    
                    # Decide action (RL Agent)
                    action = self.agent.choose_action(emotion)
                    print(f"🤖 Agent Decided: {action}")

                    if action == "voice":
                        self.soother.soothe(emotion)
                    elif action == "music":
                        self.music_player.play_music(emotion)
                    
                    # Update Agent Memory
                    reward = 1 
                    self.agent.update(emotion, action, reward, emotion)
                    self.agent.save(RL_TABLE_PATH)
                    
                    print("⏳ Soothing applied. Checking effect in 10s...")
                    next_sleep_interval = 10  # Short sleep to check if it worked

        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self):
        print("\n🛑 Shutting down system...")
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        print("👋 Goodbye.")