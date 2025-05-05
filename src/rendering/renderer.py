import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
import kaolin as kal
from kaolin.render.camera import Camera
from kaolin.render.light import DirectionalLight
from kaolin.render.mesh import Mesh
from kaolin.render.material import Material

class Renderer:
    """Handles 3D rendering and visualization of the generated worlds."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.logger = logging.getLogger("NebulaArchitect.Renderer")
        self._initialize_renderer()
        
    def _initialize_renderer(self):
        """Initialize the rendering engine."""
        try:
            # Set up camera
            self.camera = Camera.from_args(
                eye=torch.tensor([0.0, 0.0, 5.0], device=self.device),
                at=torch.tensor([0.0, 0.0, 0.0], device=self.device),
                up=torch.tensor([0.0, 1.0, 0.0], device=self.device),
                fov=60.0,
                width=self.config["rendering"]["resolution"]["width"],
                height=self.config["rendering"]["resolution"]["height"],
                device=self.device
            )
            
            # Set up default lighting
            self.lights = [
                DirectionalLight(
                    direction=torch.tensor([1.0, -1.0, 1.0], device=self.device),
                    color=torch.tensor([1.0, 1.0, 1.0], device=self.device),
                    intensity=1.0
                )
            ]
            
            self.logger.info("Rendering engine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize rendering engine: {e}")
            raise
    
    def create_mesh(self, 
                   vertices: torch.Tensor,
                   faces: torch.Tensor,
                   materials: Optional[Dict[str, Any]] = None) -> Mesh:
        """Create a mesh from vertices and faces."""
        try:
            # Create default material if none provided
            if materials is None:
                materials = {
                    "albedo": torch.ones((3,), device=self.device),
                    "roughness": 0.5,
                    "metallic": 0.0
                }
            
            # Create material
            material = Material(
                albedo=materials["albedo"],
                roughness=materials["roughness"],
                metallic=materials["metallic"],
                normal_map=materials.get("normal_map")
            )
            
            # Create mesh
            mesh = Mesh(
                vertices=vertices,
                faces=faces,
                material=material
            )
            
            return mesh
        except Exception as e:
            self.logger.error(f"Failed to create mesh: {e}")
            raise
    
    def set_camera(self, 
                  eye: torch.Tensor,
                  at: torch.Tensor,
                  up: Optional[torch.Tensor] = None):
        """Set camera position and orientation."""
        try:
            if up is None:
                up = torch.tensor([0.0, 1.0, 0.0], device=self.device)
            
            self.camera = Camera.from_args(
                eye=eye,
                at=at,
                up=up,
                fov=60.0,
                width=self.config["rendering"]["resolution"]["width"],
                height=self.config["rendering"]["resolution"]["height"],
                device=self.device
            )
        except Exception as e:
            self.logger.error(f"Failed to set camera: {e}")
            raise
    
    def add_light(self, 
                 direction: torch.Tensor,
                 color: torch.Tensor,
                 intensity: float):
        """Add a directional light to the scene."""
        try:
            light = DirectionalLight(
                direction=direction,
                color=color,
                intensity=intensity
            )
            self.lights.append(light)
        except Exception as e:
            self.logger.error(f"Failed to add light: {e}")
            raise
    
    def render_mesh(self, mesh: Mesh) -> torch.Tensor:
        """Render a mesh and return the image."""
        try:
            # Set up renderer
            renderer = kal.render.mesh.DibrRenderer(
                camera=self.camera,
                lights=self.lights,
                width=self.config["rendering"]["resolution"]["width"],
                height=self.config["rendering"]["resolution"]["height"],
                device=self.device
            )
            
            # Render mesh
            image = renderer(mesh)
            
            return image
        except Exception as e:
            self.logger.error(f"Failed to render mesh: {e}")
            raise
    
    def render_scene(self, meshes: List[Mesh]) -> torch.Tensor:
        """Render multiple meshes in a scene."""
        try:
            # Set up renderer
            renderer = kal.render.mesh.DibrRenderer(
                camera=self.camera,
                lights=self.lights,
                width=self.config["rendering"]["resolution"]["width"],
                height=self.config["rendering"]["resolution"]["height"],
                device=self.device
            )
            
            # Render all meshes
            images = []
            for mesh in meshes:
                image = renderer(mesh)
                images.append(image)
            
            # Composite images
            final_image = torch.zeros_like(images[0])
            for image in images:
                # Simple alpha compositing
                alpha = image[3:4]
                final_image = final_image * (1 - alpha) + image[:3] * alpha
            
            return final_image
        except Exception as e:
            self.logger.error(f"Failed to render scene: {e}")
            raise
    
    def save_image(self, image: torch.Tensor, filepath: str):
        """Save a rendered image to a file."""
        try:
            # Convert to numpy and save
            image_np = image.cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            kal.io.utils.save_image(filepath, image_np)
        except Exception as e:
            self.logger.error(f"Failed to save image: {e}")
            raise
    
    def cleanup(self):
        """Clean up rendering resources."""
        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info("Rendering resources cleaned up")
        except Exception as e:
            self.logger.error(f"Failed to cleanup rendering resources: {e}")
            raise 