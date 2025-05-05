import torch
import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging
from pathlib import Path

class AudioProcessor:
    """Handles audio input processing and speech recognition."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.logger = logging.getLogger("NebulaArchitect.AudioProcessor")
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize the speech recognition model."""
        try:
            self.whisper_model = whisper.load_model(
                self.config["models"]["whisper"]["model_size"],
                device=self.device
            )
            self.logger.info("Speech recognition model initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize speech recognition model: {e}")
            raise
    
    def record_audio(self, duration: float = 5.0) -> np.ndarray:
        """Record audio from the microphone."""
        try:
            self.logger.info(f"Recording audio for {duration} seconds...")
            audio_data = sd.rec(
                int(duration * self.config["audio"]["sample_rate"]),
                samplerate=self.config["audio"]["sample_rate"],
                channels=self.config["audio"]["channels"],
                dtype=self.config["audio"]["format"]
            )
            sd.wait()
            return audio_data
        except Exception as e:
            self.logger.error(f"Failed to record audio: {e}")
            raise
    
    def save_audio(self, audio_data: np.ndarray, filepath: str):
        """Save audio data to a file."""
        try:
            sf.write(filepath, audio_data, self.config["audio"]["sample_rate"])
            self.logger.info(f"Audio saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save audio: {e}")
            raise
    
    def transcribe_audio(self, audio_data: np.ndarray) -> str:
        """Transcribe audio data to text using Whisper."""
        try:
            self.logger.info("Transcribing audio...")
            result = self.whisper_model.transcribe(
                audio_data,
                language="en",
                fp16=self.config["models"]["whisper"]["compute_type"] == "float16"
            )
            return result["text"]
        except Exception as e:
            self.logger.error(f"Failed to transcribe audio: {e}")
            raise
    
    def process_voice_command(self, audio_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Process voice command and return structured data."""
        try:
            if audio_data is None:
                audio_data = self.record_audio()
            
            # Transcribe audio to text
            text = self.transcribe_audio(audio_data)
            self.logger.info(f"Transcribed text: {text}")
            
            # TODO: Implement natural language understanding
            # This would parse the text into structured commands
            
            return {
                "text": text,
                "command_type": "unknown",  # To be implemented
                "parameters": {}  # To be implemented
            }
        except Exception as e:
            self.logger.error(f"Failed to process voice command: {e}")
            raise
    
    def cleanup(self):
        """Clean up resources."""
        try:
            # Release any audio resources
            sd.stop()
            self.logger.info("Audio resources cleaned up")
        except Exception as e:
            self.logger.error(f"Failed to cleanup audio resources: {e}")
            raise 