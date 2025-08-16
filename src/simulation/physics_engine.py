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
        
        # Validate and set physics configuration with proper types
        self.physics_config = self._validate_physics_config(config)
        
        self._initialize_physics()
        self.objects: Dict[int, PhysicsObject] = {}
    
    def _validate_physics_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and set default physics configuration values."""
        try:
            # Get physics config or create default
            physics_config = config.get("models", {}).get("physics", {})
            
            # Set defaults with proper types
            defaults = {
                "timestep": 0.016,      # 1/60 seconds (float)
                "gravity": -9.81,       # gravity acceleration (float)
                "enabled": True,        # whether physics is enabled (bool)
                "solver_iterations": 10, # solver iterations (int)
                "collision_margin": 0.001, # collision margin (float)
                "gui": False,           # whether to show GUI (bool)
                "real_time": False      # whether to run in real time (bool)
            }
            
            # Validate and convert types
            validated_config = {}
            for key, default_value in defaults.items():
                if key in physics_config:
                    try:
                        # Convert to appropriate type
                        if isinstance(default_value, float):
                            validated_config[key] = float(physics_config[key])
                        elif isinstance(default_value, int):
                            validated_config[key] = int(physics_config[key])
                        elif isinstance(default_value, bool):
                            if isinstance(physics_config[key], str):
                                validated_config[key] = physics_config[key].lower() in ['true', '1', 'yes', 'on']
                            else:
                                validated_config[key] = bool(physics_config[key])
                        else:
                            validated_config[key] = physics_config[key]
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Invalid value for {key}: {physics_config[key]}, using default: {default_value}")
                        validated_config[key] = default_value
                else:
                    validated_config[key] = default_value
            
            self.logger.info(f"Physics configuration validated: {validated_config}")
            return validated_config
            
        except Exception as e:
            self.logger.error(f"Error validating physics config: {e}")
            # Return safe defaults
            return {
                "timestep": 0.016,
                "gravity": -9.81,
                "enabled": True,
                "solver_iterations": 10,
                "collision_margin": 0.001,
                "gui": False,
                "real_time": False
            }
        
    def _initialize_physics(self):
        """Initialize the physics engine."""
        try:
            # Initialize PyBullet with GUI option
            if self.physics_config.get("gui", False):
                self.client = p.connect(p.GUI)
            else:
                self.client = p.connect(p.DIRECT)  # Use DIRECT for headless mode
            
            # Set physics parameters with validated config
            p.setGravity(0, 0, self.physics_config["gravity"])
            p.setTimeStep(self.physics_config["timestep"])
            
            # Configure physics engine parameters
            p.setPhysicsEngineParameter(
                numSolverIterations=self.physics_config["solver_iterations"],
                contactBreakingThreshold=self.physics_config["collision_margin"]
            )
            
            # Set real-time simulation mode
            if self.physics_config.get("real_time", False):
                p.setRealTimeSimulation(1)
            else:
                p.setRealTimeSimulation(0)  # Disable real-time simulation for better control
            
            # Create ground plane for objects to interact with
            try:
                import pybullet_data
                p.setAdditionalSearchPath(pybullet_data.getDataPath())
                self.ground_id = p.loadURDF("plane.urdf")
                self.logger.info("Ground plane loaded successfully")
            except Exception as e:
                self.logger.warning(f"Could not load ground plane: {e}")
                # Create a simple ground plane manually
                ground_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[50, 50, 0.1])
                ground_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[50, 50, 0.1], 
                                                  rgbaColor=[0.5, 0.5, 0.5, 1])
                self.ground_id = p.createMultiBody(baseMass=0, 
                                                 baseCollisionShapeIndex=ground_collision,
                                                 baseVisualShapeIndex=ground_visual,
                                                 basePosition=[0, 0, -0.1])
            
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
                  restitution: float = 0.5,
                  size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                  color: Tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)) -> int:
        """Add a physical object to the simulation."""
        try:
            # Create collision shape with specified size
            if collision_shape == "box":
                collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in size])
            elif collision_shape == "sphere":
                radius = size[0] / 2  # Use first dimension as radius
                collision_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
            elif collision_shape == "cylinder":
                collision_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=size[0]/2, height=size[2])
            else:
                self.logger.warning(f"Unsupported collision shape: {collision_shape}, using box")
                collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in size])
            
            # Create visual shape with specified size and color
            if visual_shape == "box":
                visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in size], rgbaColor=color)
            elif visual_shape == "sphere":
                radius = size[0] / 2
                visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
            elif visual_shape == "cylinder":
                visual_id = p.createVisualShape(p.GEOM_CYLINDER, radius=size[0]/2, length=size[2], rgbaColor=color)
            else:
                self.logger.warning(f"Unsupported visual shape: {visual_shape}, using box")
                visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in size], rgbaColor=color)
            
            # Create multi-body
            body_id = p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=visual_id,
                basePosition=position.tolist() if isinstance(position, np.ndarray) else position,
                baseOrientation=orientation.tolist() if isinstance(orientation, np.ndarray) else orientation
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
                position=np.array(position),
                orientation=np.array(orientation),
                mass=mass,
                collision_shape=collision_shape,
                visual_shape=visual_shape,
                friction=friction,
                restitution=restitution
            )
            
            self.logger.info(f"Added {collision_shape} object {body_id} at position {position}")
            return body_id
        except Exception as e:
            self.logger.error(f"Failed to add object: {e}")
            raise
    
    def add_primitive(self, shape_type: str, position: Tuple[float, float, float],
                     size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                     mass: float = 1.0, 
                     color: Tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0),
                     friction: float = 0.5,
                     restitution: float = 0.5) -> int:
        """Add a primitive shape to the physics simulation."""
        try:
            position_array = np.array(position)
            orientation_array = np.array([0, 0, 0, 1])  # Default quaternion
            
            return self.add_object(
                position=position_array,
                orientation=orientation_array,
                mass=mass,
                collision_shape=shape_type.lower(),
                visual_shape=shape_type.lower(),
                friction=friction,
                restitution=restitution,
                size=size,
                color=color
            )
        except Exception as e:
            self.logger.error(f"Failed to add primitive {shape_type}: {e}")
            raise
    
    def remove_object(self, object_id: int):
        """Remove an object from the simulation."""
        try:
            if object_id in self.objects:
                p.removeBody(object_id)
                del self.objects[object_id]
                self.logger.info(f"Removed object {object_id}")
            else:
                self.logger.warning(f"Object {object_id} not found in simulation")
        except Exception as e:
            self.logger.error(f"Failed to remove object: {e}")
            raise
    
    def apply_force(self, object_id: int, force: np.ndarray, position: Optional[np.ndarray] = None):
        """Apply a force to an object."""
        try:
            if object_id not in self.objects:
                raise ValueError(f"Object {object_id} not found")
            
            if position is None:
                position = [0, 0, 0]  # Apply force at center of mass
            
            force_list = force.tolist() if isinstance(force, np.ndarray) else force
            position_list = position.tolist() if isinstance(position, np.ndarray) else position
            
            p.applyExternalForce(
                object_id,
                -1,  # Apply to base link
                force_list,
                position_list,
                p.WORLD_FRAME
            )
        except Exception as e:
            self.logger.error(f"Failed to apply force: {e}")
            raise
    
    def apply_torque(self, object_id: int, torque: np.ndarray):
        """Apply a torque to an object."""
        try:
            if object_id not in self.objects:
                raise ValueError(f"Object {object_id} not found")
            
            torque_list = torque.tolist() if isinstance(torque, np.ndarray) else torque
            
            p.applyExternalTorque(
                object_id,
                -1,  # Apply to base link
                torque_list,
                p.WORLD_FRAME
            )
        except Exception as e:
            self.logger.error(f"Failed to apply torque: {e}")
            raise
    
    def set_object_velocity(self, object_id: int, linear_velocity: Tuple[float, float, float],
                           angular_velocity: Optional[Tuple[float, float, float]] = None):
        """Set the velocity of an object."""
        try:
            if object_id not in self.objects:
                raise ValueError(f"Object {object_id} not found")
            
            if angular_velocity is None:
                angular_velocity = (0, 0, 0)
            
            p.resetBaseVelocity(object_id, linear_velocity, angular_velocity)
            
        except Exception as e:
            self.logger.error(f"Failed to set velocity for object {object_id}: {e}")
            raise
    
    def step_simulation(self, num_steps: int = 1):
        """Step the physics simulation forward."""
        try:
            for _ in range(num_steps):
                p.stepSimulation()
            
            # Update stored object positions
            self._update_object_states()
            
        except Exception as e:
            self.logger.error(f"Failed to step simulation: {e}")
            raise
    
    def _update_object_states(self):
        """Update the stored states of all physics objects."""
        try:
            for obj_id in self.objects.keys():
                if p.getNumBodies() > obj_id:
                    pos, orn = p.getBasePositionAndOrientation(obj_id)
                    self.objects[obj_id].position = np.array(pos)
                    self.objects[obj_id].orientation = np.array(orn)
        except Exception as e:
            self.logger.warning(f"Error updating object states: {e}")
    
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
                "angular_velocity": np.array(angular_velocity),
                "mass": self.objects[object_id].mass,
                "friction": self.objects[object_id].friction,
                "restitution": self.objects[object_id].restitution
            }
        except Exception as e:
            self.logger.error(f"Failed to get object state: {e}")
            raise
    
    def get_all_objects(self) -> Dict[int, Dict[str, Any]]:
        """Get states of all objects."""
        try:
            all_states = {}
            for obj_id in self.objects.keys():
                all_states[obj_id] = self.get_object_state(obj_id)
            return all_states
        except Exception as e:
            self.logger.error(f"Failed to get all object states: {e}")
            return {}
    
    def get_contacts(self, object_id: int) -> List[Dict[str, Any]]:
        """Get contact points for an object."""
        try:
            if object_id not in self.objects:
                raise ValueError(f"Object {object_id} not found")
            
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
    
    def reset_simulation(self):
        """Reset the physics simulation."""
        try:
            # Remove all objects except ground
            object_ids = list(self.objects.keys())
            for obj_id in object_ids:
                self.remove_object(obj_id)
            
            # Reset simulation
            p.resetSimulation()
            
            # Reconfigure physics
            p.setGravity(0, 0, self.physics_config["gravity"])
            p.setTimeStep(self.physics_config["timestep"])
            
            # Recreate ground plane
            try:
                import pybullet_data
                p.setAdditionalSearchPath(pybullet_data.getDataPath())
                self.ground_id = p.loadURDF("plane.urdf")
            except:
                # Fallback ground plane
                ground_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[50, 50, 0.1])
                ground_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[50, 50, 0.1], 
                                                  rgbaColor=[0.5, 0.5, 0.5, 1])
                self.ground_id = p.createMultiBody(baseMass=0, 
                                                 baseCollisionShapeIndex=ground_collision,
                                                 baseVisualShapeIndex=ground_visual,
                                                 basePosition=[0, 0, -0.1])
            
            self.logger.info("Physics simulation reset successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to reset simulation: {e}")
            raise
    
    def set_gravity(self, gravity: Tuple[float, float, float] = (0, 0, -9.81)):
        """Set gravity vector."""
        try:
            p.setGravity(gravity[0], gravity[1], gravity[2])
            self.physics_config["gravity"] = gravity[2]  # Store z-component
            self.logger.info(f"Gravity set to {gravity}")
        except Exception as e:
            self.logger.error(f"Failed to set gravity: {e}")
            raise
    
    def enable_debug_visualization(self):
        """Enable debug visualization features."""
        try:
            if self.client is not None:
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
                p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
                p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
                self.logger.info("Debug visualization enabled")
        except Exception as e:
            self.logger.error(f"Failed to enable debug visualization: {e}")
    
    def cleanup(self):
        """Clean up physics engine resources."""
        try:
            if hasattr(self, 'client') and self.client is not None:
                p.disconnect(self.client)
                self.client = None
            
            # Clear objects dictionary
            self.objects.clear()
            
            self.logger.info("Physics engine resources cleaned up")
        except Exception as e:
            self.logger.error(f"Failed to cleanup physics engine resources: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass 