import torch
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

from core.system import NebulaSystem
from audio.processor import AudioProcessor
from models.world_generator import WorldGenerator
from simulation.physics_engine import PhysicsEngine
from simulation.agent_system import AgentSystem
from rendering.renderer import Renderer

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("NebulaArchitect")

def main():
    """Main application entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="NEBULA Architect - AI-powered 3D world generation")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to run on (cuda/cpu)")
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting NEBULA Architect...")
    
    try:
        # Load configuration
        config = load_config(args.config)
        device = torch.device(args.device)
        
        # Initialize main system
        system = NebulaSystem(config, device)
        
        # Main application loop
        running = True
        while running:
            try:
                # Process voice input
                audio_processor = AudioProcessor(config, device)
                command = audio_processor.process_voice_command()
                
                if command["text"].lower() == "exit":
                    running = False
                    continue
                
                # Generate world from description
                world_generator = WorldGenerator(config, device)
                world = world_generator.generate_world(command["text"])
                
                # Set up physics simulation
                physics_engine = PhysicsEngine(config, device)
                
                # Initialize agent system
                agent_system = AgentSystem(config, device)
                
                # Set up rendering
                renderer = Renderer(config, device)
                
                # Main simulation loop
                while True:
                    # Update physics
                    physics_engine.step_simulation()
                    
                    # Update agents
                    # TODO: Implement agent behavior updates
                    
                    # Render frame
                    frame = renderer.render_scene([])  # TODO: Add meshes to render
                    
                    # Save frame
                    renderer.save_image(frame, "output/frame.png")
                    
                    # Check for exit condition
                    if not running:
                        break
                
            except KeyboardInterrupt:
                running = False
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                running = False
        
        # Cleanup
        system.cleanup()
        logger.info("NEBULA Architect shutdown complete")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main() 