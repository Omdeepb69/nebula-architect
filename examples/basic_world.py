import torch
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

from src.core.system import NebulaSystem
from src.audio.processor import AudioProcessor
from src.models.world_generator import WorldGenerator
from src.simulation.physics_engine import PhysicsEngine
from src.simulation.agent_system import AgentSystem
from src.rendering.renderer import Renderer

def main():
    """Example of creating a simple world with NEBULA Architect."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("NebulaArchitect.Example")
    
    try:
        # Load configuration
        config_path = Path("configs/config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Initialize components
        logger.info("Initializing components...")
        
        # Initialize world generator
        world_generator = WorldGenerator(config, device)
        
        # Initialize physics engine
        physics_engine = PhysicsEngine(config, device)
        
        # Initialize agent system
        agent_system = AgentSystem(config, device)
        
        # Initialize renderer
        renderer = Renderer(config, device)
        
        # Create a simple world
        logger.info("Generating world...")
        world = world_generator.generate_world(
            "A medieval castle on a hill with a moat and drawbridge"
        )
        
        # Add some physical objects
        logger.info("Adding physical objects...")
        
        # Add ground plane
        ground_id = physics_engine.add_object(
            position=[0, 0, 0],
            orientation=[0, 0, 0, 1],
            mass=0,  # Static object
            collision_shape="box",
            visual_shape="box"
        )
        
        # Add castle
        castle_id = physics_engine.add_object(
            position=[0, 5, 0],
            orientation=[0, 0, 0, 1],
            mass=1000,
            collision_shape="box",
            visual_shape="box"
        )
        
        # Add some agents
        logger.info("Adding agents...")
        
        # Add guard
        guard_id = agent_system.create_agent(
            position=[5, 0, 5],
            orientation=[0, 0, 0, 1]
        )
        
        # Add merchant
        merchant_id = agent_system.create_agent(
            position=[-5, 0, -5],
            orientation=[0, 0, 0, 1]
        )
        
        # Set up relationships
        agent_system.update_relationships(guard_id, merchant_id, 0.3)  # Slightly positive
        
        # Main simulation loop
        logger.info("Starting simulation...")
        for i in range(100):  # Simulate 100 steps
            # Update physics
            physics_engine.step_simulation()
            
            # Update agent states
            guard_state = agent_system.get_agent_state(guard_id)
            merchant_state = agent_system.get_agent_state(merchant_id)
            
            # Get nearby agents
            nearby_agents = agent_system.get_nearby_agents(
                guard_id,
                config["agents"]["interaction_radius"]
            )
            
            # Render frame
            frame = renderer.render_scene([])  # TODO: Add meshes to render
            renderer.save_image(frame, f"output/frame_{i:04d}.png")
            
            logger.info(f"Step {i}: Guard near {len(nearby_agents)} agents")
        
        # Cleanup
        logger.info("Cleaning up...")
        physics_engine.cleanup()
        agent_system.cleanup()
        renderer.cleanup()
        
        logger.info("Example completed successfully")
        
    except Exception as e:
        logger.error(f"Error in example: {e}")
        raise

if __name__ == "__main__":
    main() 