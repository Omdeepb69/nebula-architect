import torch
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Any
import os

# Import individual components instead of NebulaSystem
from audio.processor import AudioProcessor
from models.world_generator import WorldGenerator
from simulation.physics_engine import PhysicsEngine
from simulation.agent_system import AgentSystem
from rendering.renderer import Renderer

class NebulaSystem:
    """Main system coordinator for NEBULA Architect."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.logger = logging.getLogger("NebulaArchitect.System")
        
        # Initialize components
        self.audio_processor = None
        self.world_generator = None
        self.physics_engine = None
        self.agent_system = None
        self.renderer = None
        
        # Runtime state
        self.current_world = None
        self.running = False
        
        self.logger.info("NebulaSystem initialized")
    
    def initialize_components(self):
        """Initialize all system components."""
        try:
            self.logger.info("Initializing system components...")
            
            # Initialize audio processor (optional for testing)
            audio_enabled = self.config.get("audio", {}).get("enabled", False)  # Default to False
            if audio_enabled:
                self.logger.info("Audio processing enabled - this may take several minutes for first-time setup...")
                try:
                    # Set timeout for audio processor initialization
                    import signal
                    import time
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError("Audio processor initialization timed out")
                    
                    # Set 5-minute timeout
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(300)  # 5 minutes
                    
                    self.audio_processor = AudioProcessor(self.config, self.device)
                    signal.alarm(0)  # Cancel timeout
                    self.logger.info("Audio processor initialized")
                    
                except (TimeoutError, Exception) as e:
                    signal.alarm(0)  # Cancel timeout
                    self.logger.warning(f"Audio processor failed to initialize: {e}")
                    self.logger.info("Continuing without audio processing - using text input fallback")
                    self.audio_processor = None
            else:
                self.logger.info("Audio processing disabled - using text input only")
                self.audio_processor = None
            
            # Initialize world generator
            self.world_generator = WorldGenerator(self.config, self.device)
            self.logger.info("World generator initialized")
            
            # Initialize physics engine
            self.physics_engine = PhysicsEngine(self.config, self.device)
            self.logger.info("Physics engine initialized")
            
            # Initialize agent system
            self.agent_system = AgentSystem(self.config, self.device)
            self.logger.info("Agent system initialized")
            
            # Initialize renderer
            self.renderer = Renderer(self.config, self.device)
            self.logger.info("Renderer initialized")
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def process_voice_command(self) -> Dict[str, Any]:
        """Process voice command input."""
        if self.audio_processor is None:
            # Return text input as fallback
            command_text = input("Enter command (or 'exit' to quit): ")
            return {"text": command_text, "confidence": 1.0}
        
        return self.audio_processor.process_voice_command()
    
    def generate_world(self, description: str):
        """Generate a world from description."""
        if self.world_generator is None:
            raise RuntimeError("World generator not initialized")
        
        self.current_world = self.world_generator.generate_world(description)
        return self.current_world
    
    def run_simulation_step(self):
        """Run a single simulation step."""
        try:
            # Update physics
            if self.physics_engine:
                self.physics_engine.step_simulation()
            
            # Update agents
            if self.agent_system and self.current_world:
                # Update agent behaviors based on world state
                pass  # TODO: Implement agent updates
            
            # Render frame
            if self.renderer and self.current_world:
                meshes = getattr(self.current_world, 'meshes', [])
                frame = self.renderer.render_scene(meshes)
                return frame
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in simulation step: {e}")
            return None
    
    def save_frame(self, frame: torch.Tensor, output_dir: str, frame_number: int):
        """Save a rendered frame."""
        if frame is None or self.renderer is None:
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save frame
        filename = os.path.join(output_dir, f"frame_{frame_number:06d}.png")
        self.renderer.save_image(frame, filename)
    
    def cleanup(self):
        """Clean up all system resources."""
        try:
            if self.renderer:
                self.renderer.cleanup()
            
            if self.physics_engine:
                self.physics_engine.cleanup()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("System cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set default values if config is incomplete
        if config is None:
            config = {}
        
        # Ensure required sections exist
        config.setdefault("rendering", {
            "resolution": {"width": 800, "height": 600},
            "smooth_meshes": False,
            "subdivide_meshes": False
        })
        
        config.setdefault("audio", {"enabled": True})
        config.setdefault("physics", {"enabled": True})
        config.setdefault("agents", {"enabled": True})
        
        # Ensure physics configuration with proper types
        config.setdefault("models", {})
        config["models"].setdefault("physics", {
            "timestep": 0.016,  # 1/60 seconds as float
            "gravity": -9.81,   # gravity as float
            "enabled": True
        })
        
        # Convert string values to proper types if they exist
        if "models" in config and "physics" in config["models"]:
            physics_config = config["models"]["physics"]
            
            # Convert timestep to float if it's a string
            if "timestep" in physics_config and isinstance(physics_config["timestep"], str):
                try:
                    physics_config["timestep"] = float(physics_config["timestep"])
                except ValueError:
                    physics_config["timestep"] = 0.016  # Default value
            
            # Convert gravity to float if it's a string
            if "gravity" in physics_config and isinstance(physics_config["gravity"], str):
                try:
                    physics_config["gravity"] = float(physics_config["gravity"])
                except ValueError:
                    physics_config["gravity"] = -9.81  # Default value
        
        return config
        
    except FileNotFoundError:
        # Create default config if file doesn't exist
        default_config = {
            "rendering": {
                "resolution": {"width": 800, "height": 600},
                "smooth_meshes": False,
                "subdivide_meshes": False
            },
            "audio": {"enabled": False},  # Disabled by default to prevent hanging
            "physics": {"enabled": True},
            "agents": {"enabled": True},
            "world_generation": {"enabled": True},
            "models": {
                "physics": {
                    "timestep": 0.016,  # 1/60 seconds
                    "gravity": -9.81,   # gravity acceleration
                    "enabled": True,
                    "solver_iterations": 10,
                    "collision_margin": 0.001
                },
                "world": {
                    "max_objects": 100,
                    "terrain_size": 50.0,
                    "seed": 42
                },
                "agent": {
                    "max_agents": 10,
                    "update_frequency": 30.0
                }
            }
        }
        
        # Try to create the config file
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.safe_dump(default_config, f, default_flow_style=False)
            print(f"Created default config file at {config_path}")
        except Exception as e:
            print(f"Could not create config file: {e}")
        
        return default_config
    
    except Exception as e:
        print(f"Error loading config: {e}")
        return {
            "rendering": {
                "resolution": {"width": 800, "height": 600}
            }
        }

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('nebula_architect.log')
        ]
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
    parser.add_argument("--output-dir", type=str, default="output",
                      help="Output directory for rendered frames")
    parser.add_argument("--interactive", action="store_true", default=True,
                      help="Run in interactive mode")
    parser.add_argument("--no-audio", action="store_true", default=False,
                      help="Disable audio processing (use text input only)")
    parser.add_argument("--test-mode", action="store_true", default=False,
                      help="Run in test mode with minimal components")
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting NEBULA Architect...")
    
    try:
        # Load configuration
        config = load_config(args.config)
        device = torch.device(args.device)
        
        # Override audio setting based on command line arguments
        if args.no_audio or args.test_mode:
            config["audio"]["enabled"] = False
            logger.info("Audio processing disabled via command line")
        
        logger.info(f"Using device: {device}")
        logger.info(f"Config loaded from: {args.config}")
        
        # Initialize main system
        system = NebulaSystem(config, device)
        system.initialize_components()
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Main application loop
        running = True
        frame_number = 0
        
        while running:
            try:
                logger.info("Waiting for voice command...")
                
                # Process voice input or text input
                command = system.process_voice_command()
                command_text = command["text"].strip().lower()
                
                if command_text in ["exit", "quit", "stop"]:
                    logger.info("Exit command received")
                    running = False
                    continue
                
                if not command_text:
                    continue
                
                logger.info(f"Processing command: {command_text}")
                
                # Generate world from description
                try:
                    world = system.generate_world(command_text)
                    logger.info("World generated successfully")
                except Exception as e:
                    logger.error(f"Failed to generate world: {e}")
                    continue
                
                # Interactive mode - run simulation loop
                if args.interactive:
                    simulation_running = True
                    steps = 0
                    max_steps = 100  # Prevent infinite loops
                    
                    logger.info("Starting simulation loop...")
                    
                    while simulation_running and steps < max_steps:
                        try:
                            # Run simulation step
                            frame = system.run_simulation_step()
                            
                            if frame is not None:
                                # Save frame
                                system.save_frame(frame, args.output_dir, frame_number)
                                frame_number += 1
                                
                                if frame_number % 10 == 0:
                                    logger.info(f"Rendered {frame_number} frames")
                            
                            steps += 1
                            
                            # Check for new commands (non-blocking)
                            # For now, just run for a fixed number of steps
                            if steps >= 10:  # Render 10 frames then wait for new command
                                simulation_running = False
                            
                        except KeyboardInterrupt:
                            logger.info("Simulation interrupted by user")
                            simulation_running = False
                        except Exception as e:
                            logger.error(f"Error in simulation loop: {e}")
                            simulation_running = False
                    
                    logger.info(f"Simulation completed. Rendered {steps} frames.")
                
                else:
                    # Non-interactive mode - render single frame
                    frame = system.run_simulation_step()
                    if frame is not None:
                        system.save_frame(frame, args.output_dir, frame_number)
                        frame_number += 1
                        logger.info(f"Single frame rendered and saved")
                
            except KeyboardInterrupt:
                logger.info("Application interrupted by user")
                running = False
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                if not args.interactive:
                    running = False
                else:
                    logger.info("Continuing in interactive mode...")
        
        # Cleanup
        logger.info("Shutting down system...")
        system.cleanup()
        logger.info("NEBULA Architect shutdown complete")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()