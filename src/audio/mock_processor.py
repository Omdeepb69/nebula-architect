# Option 1: Create a simple mock AudioProcessor for testing
# Save this as src/audio/mock_processor.py

import torch
import logging
from typing import Dict, Any

class MockAudioProcessor:
    """Mock audio processor for testing without Whisper dependencies."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.logger = logging.getLogger("NebulaArchitect.MockAudioProcessor")
        self.logger.info("Mock audio processor initialized (text input only)")
    
    def process_voice_command(self) -> Dict[str, Any]:
        """Process voice command by falling back to text input."""
        try:
            command_text = input("Enter command (or 'exit' to quit): ")
            return {
                "text": command_text,
                "confidence": 1.0,
                "language": "en"
            }
        except (EOFError, KeyboardInterrupt):
            return {"text": "exit", "confidence": 1.0}
        except Exception as e:
            self.logger.error(f"Error in mock audio processing: {e}")
            return {"text": "", "confidence": 0.0}
    
    def cleanup(self):
        """Cleanup mock processor."""
        pass

# Option 2: Quick config.yaml content
"""
Create configs/config.yaml with this content to skip audio:

rendering:
  resolution:
    width: 800
    height: 600
  smooth_meshes: false
  subdivide_meshes: false

audio:
  enabled: false  # This prevents Whisper loading

physics:
  enabled: true

agents:
  enabled: true

world_generation:
  enabled: true

models:
  physics:
    timestep: 0.016
    gravity: -9.81
    enabled: true
    solver_iterations: 10
    collision_margin: 0.001
    gui: false
    real_time: false
  world:
    max_objects: 100
    terrain_size: 50.0
    seed: 42
  agent:
    max_agents: 10
    update_frequency: 30.0
"""

# Option 3: Modify the main.py import to use mock
# Add this at the top of main.py after the existing imports:

"""
# Replace the AudioProcessor import with this conditional import:
try:
    from audio.processor import AudioProcessor
except ImportError:
    # Fallback to mock if real processor unavailable
    class AudioProcessor:
        def __init__(self, config, device):
            self.logger = logging.getLogger("MockAudio")
            self.logger.info("Using text input fallback")
        def process_voice_command(self):
            text = input("Enter command: ")
            return {"text": text, "confidence": 1.0}
"""