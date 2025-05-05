import torch
import numpy as np
import pybullet as p
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass

@dataclass
class PhysicsObject:
    """Represents a physical object in the simulation."""
    id: int
    position: np.ndarray
    orientation: np.ndarray
    mass: float
    collision_shape: str
    visual_shape: str
    friction: float
    restitution: float

class PhysicsEngine:
    """Handles physical simulations and interactions in the 3D world."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.logger = logging.getLogger("NebulaArchitect.PhysicsEngine")
        self._initialize_physics()
        self.objects: Dict[int, PhysicsObject] = {}
        
    def _initialize_physics(self):
        """Initialize the physics engine."""
        try:
            # Initialize PyBullet
            self.client = p.connect(p.DIRECT)  # Use DIRECT for headless mode
            p.setGravity(0, 0, self.config["models"]["physics"]["gravity"])
            p.setTimeStep(self.config["models"]["physics"]["timestep"])
            p.setRealTimeSimulation(0)  # Disable real-time simulation for better control
            
            self.logger.info("Physics engine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize physics engine: {e}")
            raise
    
    def add_object(self, 
                  position: np.ndarray,
                  orientation: np.ndarray,
                  mass: float,
                  collision_shape: str,
                  visual_shape: str,
                  friction: float = 0.5,
                  restitution: float = 0.5) -> int:
        """Add a physical object to the simulation."""
        try:
            # Create collision shape
            if collision_shape == "box":
                collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
            elif collision_shape == "sphere":
                collision_id = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5)
            elif collision_shape == "cylinder":
                collision_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.5, height=1.0)
            else:
                raise ValueError(f"Unsupported collision shape: {collision_shape}")
            
            # Create visual shape
            if visual_shape == "box":
                visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
            elif visual_shape == "sphere":
                visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.5)
            elif visual_shape == "cylinder":
                visual_id = p.createVisualShape(p.GEOM_CYLINDER, radius=0.5, length=1.0)
            else:
                raise ValueError(f"Unsupported visual shape: {visual_shape}")
            
            # Create multi-body
            body_id = p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=visual_id,
                basePosition=position,
                baseOrientation=orientation
            )
            
            # Set friction and restitution
            p.changeDynamics(
                body_id,
                -1,
                lateralFriction=friction,
                restitution=restitution
            )
            
            # Store object information
            self.objects[body_id] = PhysicsObject(
                id=body_id,
                position=position,
                orientation=orientation,
                mass=mass,
                collision_shape=collision_shape,
                visual_shape=visual_shape,
                friction=friction,
                restitution=restitution
            )
            
            return body_id
        except Exception as e:
            self.logger.error(f"Failed to add object: {e}")
            raise
    
    def remove_object(self, object_id: int):
        """Remove an object from the simulation."""
        try:
            if object_id in self.objects:
                p.removeBody(object_id)
                del self.objects[object_id]
        except Exception as e:
            self.logger.error(f"Failed to remove object: {e}")
            raise
    
    def apply_force(self, object_id: int, force: np.ndarray, position: Optional[np.ndarray] = None):
        """Apply a force to an object."""
        try:
            if position is None:
                position = [0, 0, 0]  # Apply force at center of mass
            p.applyExternalForce(
                object_id,
                -1,  # Apply to base link
                force,
                position,
                p.WORLD_FRAME
            )
        except Exception as e:
            self.logger.error(f"Failed to apply force: {e}")
            raise
    
    def apply_torque(self, object_id: int, torque: np.ndarray):
        """Apply a torque to an object."""
        try:
            p.applyExternalTorque(
                object_id,
                -1,  # Apply to base link
                torque,
                p.WORLD_FRAME
            )
        except Exception as e:
            self.logger.error(f"Failed to apply torque: {e}")
            raise
    
    def step_simulation(self, num_steps: int = 1):
        """Step the physics simulation forward."""
        try:
            for _ in range(num_steps):
                p.stepSimulation()
        except Exception as e:
            self.logger.error(f"Failed to step simulation: {e}")
            raise
    
    def get_object_state(self, object_id: int) -> Dict[str, Any]:
        """Get the current state of an object."""
        try:
            if object_id not in self.objects:
                raise ValueError(f"Object {object_id} not found")
            
            position, orientation = p.getBasePositionAndOrientation(object_id)
            linear_velocity, angular_velocity = p.getBaseVelocity(object_id)
            
            return {
                "position": np.array(position),
                "orientation": np.array(orientation),
                "linear_velocity": np.array(linear_velocity),
                "angular_velocity": np.array(angular_velocity)
            }
        except Exception as e:
            self.logger.error(f"Failed to get object state: {e}")
            raise
    
    def get_contacts(self, object_id: int) -> List[Dict[str, Any]]:
        """Get contact points for an object."""
        try:
            contacts = p.getContactPoints(object_id)
            return [
                {
                    "contact_point": np.array(contact[5]),
                    "contact_normal": np.array(contact[7]),
                    "contact_force": contact[9],
                    "contact_object": contact[2]
                }
                for contact in contacts
            ]
        except Exception as e:
            self.logger.error(f"Failed to get contacts: {e}")
            raise
    
    def cleanup(self):
        """Clean up physics engine resources."""
        try:
            p.disconnect(self.client)
            self.logger.info("Physics engine resources cleaned up")
        except Exception as e:
            self.logger.error(f"Failed to cleanup physics engine resources: {e}")
            raise 