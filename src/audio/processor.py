import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging
from pathlib import Path
import time

# Try to import audio libraries with fallbacks
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Whisper not available: {e}. Using fallback speech recognition.")
    WHISPER_AVAILABLE = False

try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_RECORDING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Audio recording libraries not available: {e}. Using text input fallback.")
    AUDIO_RECORDING_AVAILABLE = False

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

class AudioProcessor:
    """Handles audio input processing and speech recognition with multiple fallbacks."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.logger = logging.getLogger("NebulaArchitect.AudioProcessor")
        
        # Audio configuration with defaults
        self.audio_config = config.get("audio", {
            "sample_rate": 16000,
            "channels": 1,
            "format": "float32",
            "chunk_size": 1024,
            "record_duration": 5.0
        })
        
        # Model configuration with defaults
        self.model_config = config.get("models", {}).get("whisper", {
            "model_size": "base",
            "compute_type": "float32"
        })
        
        self.whisper_model = None
        self.speech_recognizer = None
        
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize available speech recognition models."""
        initialized = False
        
        # Try to initialize Whisper first
        if WHISPER_AVAILABLE:
            try:
                self.logger.info("Initializing Whisper model...")
                self.whisper_model = whisper.load_model(
                    self.model_config["model_size"],
                    device=self.device if self.device.type == "cuda" else "cpu"
                )
                self.logger.info("Whisper model initialized successfully")
                initialized = True
            except Exception as e:
                self.logger.error(f"Failed to initialize Whisper model: {e}")
                self.whisper_model = None
        
        # Fallback to speech_recognition library
        if not initialized and SPEECH_RECOGNITION_AVAILABLE:
            try:
                self.logger.info("Initializing SpeechRecognition fallback...")
                self.speech_recognizer = sr.Recognizer()
                self.logger.info("SpeechRecognition initialized successfully")
                initialized = True
            except Exception as e:
                self.logger.error(f"Failed to initialize SpeechRecognition: {e}")
                self.speech_recognizer = None
        
        if not initialized:
            self.logger.warning("No speech recognition models available. Will use text input fallback.")
    
    def record_audio(self, duration: Optional[float] = None) -> Optional[np.ndarray]:
        """Record audio from the microphone."""
        if not AUDIO_RECORDING_AVAILABLE:
            self.logger.warning("Audio recording not available")
            return None
            
        try:
            duration = duration or self.audio_config["record_duration"]
            self.logger.info(f"Recording audio for {duration} seconds...")
            
            # Record audio
            audio_data = sd.rec(
                int(duration * self.audio_config["sample_rate"]),
                samplerate=self.audio_config["sample_rate"],
                channels=self.audio_config["channels"],
                dtype=self.audio_config["format"]
            )
            sd.wait()  # Wait until recording is finished
            
            # Convert to the format expected by Whisper (mono, float32)
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)  # Convert to mono
            
            return audio_data.flatten()
            
        except Exception as e:
            self.logger.error(f"Failed to record audio: {e}")
            return None
    
    def save_audio(self, audio_data: np.ndarray, filepath: str) -> bool:
        """Save audio data to a file."""
        try:
            sf.write(filepath, audio_data, self.audio_config["sample_rate"])
            self.logger.info(f"Audio saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save audio: {e}")
            return False
    
    def transcribe_audio_whisper(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio data to text using Whisper."""
        if self.whisper_model is None:
            return None
            
        try:
            self.logger.info("Transcribing audio with Whisper...")
            
            # Ensure audio is in the right format for Whisper
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Whisper expects audio to be normalized to [-1, 1]
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            result = self.whisper_model.transcribe(
                audio_data,
                language="en",
                fp16=self.model_config["compute_type"] == "float16"
            )
            
            text = result["text"].strip()
            self.logger.info(f"Whisper transcription: {text}")
            return text
            
        except Exception as e:
            self.logger.error(f"Failed to transcribe audio with Whisper: {e}")
            return None
    
    def transcribe_audio_fallback(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio using speech_recognition library as fallback."""
        if self.speech_recognizer is None or not AUDIO_RECORDING_AVAILABLE:
            return None
            
        try:
            self.logger.info("Transcribing audio with SpeechRecognition fallback...")
            
            # Convert numpy array to AudioData
            # Save to temporary wav file first
            temp_path = "temp_audio.wav"
            sf.write(temp_path, audio_data, self.audio_config["sample_rate"])
            
            # Load with speech_recognition
            with sr.AudioFile(temp_path) as source:
                audio = self.speech_recognizer.record(source)
            
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)
            
            # Try different recognition services
            text = None
            try:
                text = self.speech_recognizer.recognize_google(audio, language="en-US")
            except sr.UnknownValueError:
                self.logger.warning("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                self.logger.warning(f"Could not request results from Google Speech Recognition: {e}")
            
            if text:
                self.logger.info(f"SpeechRecognition transcription: {text}")
                return text
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to transcribe audio with fallback: {e}")
            return None
    
    def transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio data to text using available methods."""
        # Try Whisper first
        if self.whisper_model is not None:
            text = self.transcribe_audio_whisper(audio_data)
            if text:
                return text
        
        # Fallback to speech_recognition
        if self.speech_recognizer is not None:
            text = self.transcribe_audio_fallback(audio_data)
            if text:
                return text
        
        self.logger.warning("No speech recognition method available")
        return None
    
    def get_text_input_fallback(self) -> str:
        """Get text input from user when audio is not available."""
        try:
            print("\n" + "="*50)
            print("NEBULA Architect - Voice Command Interface")
            print("(Audio processing not available - using text input)")
            print("="*50)
            print("Enter your world description (or 'exit' to quit):")
            text = input("> ").strip()
            return text
        except KeyboardInterrupt:
            return "exit"
        except Exception as e:
            self.logger.error(f"Error getting text input: {e}")
            return "exit"
    
    def process_voice_command(self, audio_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Process voice command and return structured data."""
        try:
            text = None
            
            # If no audio data provided, try to record
            if audio_data is None and AUDIO_RECORDING_AVAILABLE:
                self.logger.info("No audio data provided, attempting to record...")
                audio_data = self.record_audio()
            
            # Try to transcribe audio
            if audio_data is not None:
                text = self.transcribe_audio(audio_data)
            
            # Fallback to text input if audio processing fails
            if not text:
                self.logger.info("Audio processing failed or unavailable, using text input...")
                text = self.get_text_input_fallback()
            
            if not text:
                text = "exit"  # Default if all methods fail
            
            # Parse command type (basic implementation)
            command_type = self._parse_command_type(text)
            
            # Extract parameters (basic implementation)
            parameters = self._extract_parameters(text)
            
            result = {
                "text": text,
                "command_type": command_type,
                "parameters": parameters,
                "timestamp": time.time(),
                "method": "audio" if audio_data is not None else "text"
            }
            
            self.logger.info(f"Processed command: {result['text'][:100]}...")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process voice command: {e}")
            return {
                "text": "exit",
                "command_type": "exit",
                "parameters": {},
                "timestamp": time.time(),
                "method": "error"
            }
    
    def _parse_command_type(self, text: str) -> str:
        """Parse command type from text (basic implementation)."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["exit", "quit", "stop", "bye"]):
            return "exit"
        elif any(word in text_lower for word in ["create", "generate", "build", "make"]):
            return "generate_world"
        elif any(word in text_lower for word in ["modify", "change", "update", "edit"]):
            return "modify_world"
        elif any(word in text_lower for word in ["save", "export"]):
            return "save_world"
        elif any(word in text_lower for word in ["load", "import", "open"]):
            return "load_world"
        else:
            return "generate_world"  # Default to world generation
    
    def _extract_parameters(self, text: str) -> Dict[str, Any]:
        """Extract parameters from text (basic implementation)."""
        parameters = {}
        
        # Extract basic parameters
        text_lower = text.lower()
        
        # Size indicators
        if any(word in text_lower for word in ["large", "big", "huge", "massive"]):
            parameters["size"] = "large"
        elif any(word in text_lower for word in ["small", "tiny", "mini"]):
            parameters["size"] = "small"
        else:
            parameters["size"] = "medium"
        
        # Style indicators
        if any(word in text_lower for word in ["realistic", "photorealistic"]):
            parameters["style"] = "realistic"
        elif any(word in text_lower for word in ["cartoon", "stylized", "anime"]):
            parameters["style"] = "stylized"
        elif any(word in text_lower for word in ["fantasy", "magical", "mystical"]):
            parameters["style"] = "fantasy"
        
        # Time of day
        if any(word in text_lower for word in ["night", "dark", "evening"]):
            parameters["time_of_day"] = "night"
        elif any(word in text_lower for word in ["morning", "dawn", "sunrise"]):
            parameters["time_of_day"] = "morning"
        elif any(word in text_lower for word in ["afternoon", "day", "noon"]):
            parameters["time_of_day"] = "day"
        elif any(word in text_lower for word in ["sunset", "dusk", "twilight"]):
            parameters["time_of_day"] = "sunset"
        
        return parameters
    
    def test_audio_system(self) -> Dict[str, bool]:
        """Test the audio system components."""
        results = {
            "whisper_available": WHISPER_AVAILABLE,
            "audio_recording_available": AUDIO_RECORDING_AVAILABLE,
            "speech_recognition_available": SPEECH_RECOGNITION_AVAILABLE,
            "whisper_model_loaded": self.whisper_model is not None,
            "speech_recognizer_loaded": self.speech_recognizer is not None
        }
        
        self.logger.info("Audio system test results:")
        for component, status in results.items():
            self.logger.info(f"  {component}: {'✓' if status else '✗'}")
        
        return results
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if AUDIO_RECORDING_AVAILABLE:
                sd.stop()
            
            # Clear models from memory
            if self.whisper_model is not None:
                del self.whisper_model
                self.whisper_model = None
            
            if self.speech_recognizer is not None:
                del self.speech_recognizer
                self.speech_recognizer = None
            
            # Clear GPU cache if using CUDA
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            self.logger.info("Audio resources cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup audio resources: {e}")