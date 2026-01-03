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
        
        try:
            while self.running:
                # Polling interval (don't check too fast)
                time.sleep(60) 
                
                # Get Audio
                segment = self.audio_buffer.get_audio_segment()
                if len(segment) < SEGMENT_SIZE:
                    continue

                # 1. Predict Emotion
                emotion, confidence = self.cry_classifier.predict(segment)
                posture = self._detect_posture()
                
                # Only log/act if it's not silence or if confidence is high enough
                if emotion != "silence":
                    print(f"🔍 Analysis: {emotion} ({confidence:.2f}) | Posture: {posture}")

                    # 2. Send Data to Flutter App
                    self.ws_server.broadcast_data({
                        "emotion": emotion,
                        "confidence": confidence,
                        "posture": posture
                    })

                    # 3. RL Decision & Action
                    # Decide action based on emotion
                    action = self.agent.choose_action(emotion)
                    print(f"🤖 Agent Chose: {action}")

                    if action == "voice":
                        self.soother.soothe(emotion)
                    elif action == "music":
                        self.music_player.play_music(emotion)
                    
                    # 4. Update Agent
                    # Simple reward logic: 1 if we did something (placeholder)
                    # Real reward should be based on if the baby STOPS crying in the next loop
                    reward = 1 
                    self.agent.update(emotion, action, reward, emotion)
                    self.agent.save(RL_TABLE_PATH)
                    
                    # Wait after action to let it have effect (so we don't spam speech)
                    print("⏳ Waiting for soothing effect...")
                    time.sleep(60) 
                    
                    # Clear buffer effectively (by waiting) or manual clear if needed
                    # For deque, old data slides out, so waiting is usually enough.

        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self):
        print("\n🛑 Shutting down system...")
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        print("👋 Goodbye.")