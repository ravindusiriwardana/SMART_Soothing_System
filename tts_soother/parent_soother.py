import os
from .services import LLMService, TTSService

class ParentSoother:
    def __init__(self, llm_model, tts_model, parent_name, parent_voice_path):
        self.parent_voice_path = parent_voice_path
        self.processed_voice_path = "processed_parent.wav"
        
        # Composition: Soother HAS-A LLMService and TTSService
        self.llm_service = LLMService(llm_model, parent_name)
        self.tts_service = TTSService(tts_model)

        # Pre-process voice once at startup if available
        if self.parent_voice_path and os.path.exists(self.parent_voice_path):
            self.tts_service.preprocess_voice(self.parent_voice_path, self.processed_voice_path)

    def soothe(self, emotion):
        """Orchestrates the soothing process."""
        # 1. Generate Text
        phrase = self.llm_service.generate_phrase(emotion)
        print(f"📝 Generated phrase: {phrase}")

        # 2. Synthesize and Play
        # Use processed voice if available, otherwise raw path (or None)
        voice_file = self.processed_voice_path if os.path.exists(self.processed_voice_path) else self.parent_voice_path
        
        self.tts_service.synthesize_and_play(
            text=phrase, 
            output_file="output_soothe.wav", 
            speaker_wav=voice_file
        )