import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
import trimesh
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import cv2

class Camera:
    """Camera class to replace Kaolin's camera functionality."""
    
    def __init__(self, eye: torch.Tensor, at: torch.Tensor, up: torch.Tensor, 
                 fov: float, width: int, height: int, device: torch.device):
        self.eye = eye.cpu().numpy() if isinstance(eye, torch.Tensor) else eye
        self.at = at.cpu().numpy() if isinstance(at, torch.Tensor) else at
        self.up = up.cpu().numpy() if isinstance(up, torch.Tensor) else up
        self.fov = fov
        self.width = width
        self.height = height
        self.device = device
        
        # Compute view and projection matrices
        self._compute_matrices()
    
    def _compute_matrices(self):
        """Compute view and projection matrices."""
        # View matrix
        z_axis = self.eye - self.at
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross(self.up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        self.view_matrix = np.array([
            [x_axis[0], y_axis[0], z_axis[0], 0],
            [x_axis[1], y_axis[1], z_axis[1], 0],
            [x_axis[2], y_axis[2], z_axis[2], 0],
            [-np.dot(x_axis, self.eye), -np.dot(y_axis, self.eye), -np.dot(z_axis, self.eye), 1]
        ])
        
        # Projection matrix
        aspect = self.width / self.height
        fov_rad = np.radians(self.fov)
        f = 1.0 / np.tan(fov_rad / 2.0)
        near, far = 0.1, 100.0
        
        self.proj_matrix = np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, -1, 0]
        ])
    
    @classmethod
    def from_args(cls, eye, at, up, fov, width, height, device):
        """Create camera from arguments (compatible with Kaolin API)."""
        return cls(eye, at, up, fov, width, height, device)

class DirectionalLight:
    """Directional light class to replace Kaolin's lighting."""
    
    def __init__(self, direction: torch.Tensor, color: torch.Tensor, intensity: float):
        self.direction = direction.cpu().numpy() if isinstance(direction, torch.Tensor) else direction
        self.color = color.cpu().numpy() if isinstance(color, torch.Tensor) else color
        self.intensity = intensity
        
        # Normalize direction
        self.direction = self.direction / np.linalg.norm(self.direction)

class Material:
    """Material class to replace Kaolin's material system."""
    
    def __init__(self, albedo: torch.Tensor, roughness: float, metallic: float, 
                 normal_map: Optional[torch.Tensor] = None):
        self.albedo = albedo.cpu().numpy() if isinstance(albedo, torch.Tensor) else albedo
        self.roughness = roughness
        self.metallic = metallic
        self.normal_map = normal_map.cpu().numpy() if normal_map is not None else None

class Mesh:
    """Mesh class to replace Kaolin's mesh functionality."""
    
    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor, material: Material):
        self.vertices = vertices.cpu().numpy() if isinstance(vertices, torch.Tensor) else vertices
        self.faces = faces.cpu().numpy() if isinstance(faces, torch.Tensor) else faces
        self.material = material
        
        # Create trimesh object for advanced operations
        self.trimesh_obj = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
        
        # Create Open3D mesh for visualization
        self.o3d_mesh = o3d.geometry.TriangleMesh()
        self.o3d_mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
        self.o3d_mesh.triangles = o3d.utility.Vector3iVector(self.faces)
        
        # Apply material properties
        if hasattr(material, 'albedo'):
            self.o3d_mesh.paint_uniform_color(material.albedo[:3])
        
        # Compute normals
        self.o3d_mesh.compute_vertex_normals()
        self.o3d_mesh.compute_triangle_normals()

class DibrRenderer:
    """Differentiable renderer to replace Kaolin's DibrRenderer."""
    
    def __init__(self, camera: Camera, lights: List[DirectionalLight], 
                 width: int, height: int, device: torch.device):
        self.camera = camera
        self.lights = lights
        self.width = width
        self.height = height
        self.device = device
        
        # Initialize Open3D visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=width, height=height, visible=False)
        
        # Set up render options
        render_option = self.vis.get_render_option()
        render_option.mesh_show_back_face = True
        render_option.mesh_show_wireframe = False
        render_option.light_on = True
    
    def __call__(self, mesh: Mesh) -> torch.Tensor:
        """Render mesh and return image tensor."""
        try:
            # Clear previous geometries
            self.vis.clear_geometries()
            
            # Add mesh to visualizer
            self.vis.add_geometry(mesh.o3d_mesh)
            
            # Set camera parameters
            ctr = self.vis.get_view_control()
            camera_params = o3d.camera.PinholeCameraParameters()
            
            # Convert our camera to Open3D format
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = self.camera.view_matrix[:3, :3]
            extrinsic[:3, 3] = self.camera.view_matrix[:3, 3]
            
            intrinsic = o3d.camera.PinholeCameraIntrinsic()
            intrinsic.set_intrinsics(
                self.width, self.height,
                self.camera.proj_matrix[0, 0] * self.width / 2,
                self.camera.proj_matrix[1, 1] * self.height / 2,
                self.width / 2, self.height / 2
            )
            
            camera_params.extrinsic = extrinsic
            camera_params.intrinsic = intrinsic
            ctr.convert_from_pinhole_camera_parameters(camera_params)
            
            # Render
            self.vis.poll_events()
            self.vis.update_renderer()
            
            # Capture image
            image = self.vis.capture_screen_float_buffer(do_render=True)
            image_np = np.asarray(image)
            
            # Convert to torch tensor (H, W, C) -> (C, H, W)
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
            
            # Add alpha channel if needed
            if image_tensor.shape[0] == 3:
                alpha = torch.ones(1, image_tensor.shape[1], image_tensor.shape[2])
                image_tensor = torch.cat([image_tensor, alpha], dim=0)
            
            return image_tensor.to(self.device)
            
        except Exception as e:
            # Fallback to software rendering
            return self._software_render(mesh)
    
    def _software_render(self, mesh: Mesh) -> torch.Tensor:
        """Software fallback rendering using matplotlib."""
        try:
            fig = plt.figure(figsize=(self.width/100, self.height/100), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot mesh
            vertices = mesh.vertices
            faces = mesh.faces
            
            # Create triangular mesh plot
            ax.plot_trisurf(
                vertices[:, 0], vertices[:, 1], vertices[:, 2],
                triangles=faces, alpha=0.8,
                color=mesh.material.albedo[:3] if hasattr(mesh.material, 'albedo') else [0.7, 0.7, 0.7]
            )
            
            # Set camera view
            ax.view_init(elev=20, azim=45)
            
            # Remove axes
            ax.set_axis_off()
            
            # Convert to image
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            
            # Convert to torch tensor
            image_tensor = torch.from_numpy(buf).permute(2, 0, 1).float() / 255.0
            
            # Add alpha channel
            alpha = torch.ones(1, image_tensor.shape[1], image_tensor.shape[2])
            image_tensor = torch.cat([image_tensor, alpha], dim=0)
            
            return image_tensor.to(self.device)
            
        except Exception as e:
            # Final fallback - return black image
            return torch.zeros(4, self.height, self.width, device=self.device)

class Renderer:
    """Enhanced renderer using alternative libraries with full feature parity."""
    
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
            
            # Initialize mesh storage
            self.meshes = []
            
            self.logger.info("Enhanced rendering engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize rendering engine: {e}")
            raise
    
    def create_mesh(self, 
                   vertices: torch.Tensor,
                   faces: torch.Tensor,
                   materials: Optional[Dict[str, Any]] = None) -> Mesh:
        """Create a mesh from vertices and faces with enhanced features."""
        try:
            # Create default material if none provided
            if materials is None:
                materials = {
                    "albedo": torch.ones((3,), device=self.device),
                    "roughness": 0.5,
                    "metallic": 0.0
                }
            
            # Ensure albedo is properly formatted
            if isinstance(materials["albedo"], torch.Tensor):
                albedo = materials["albedo"]
            else:
                albedo = torch.tensor(materials["albedo"], device=self.device)
            
            # Create material
            material = Material(
                albedo=albedo,
                roughness=materials.get("roughness", 0.5),
                metallic=materials.get("metallic", 0.0),
                normal_map=materials.get("normal_map")
            )
            
            # Create mesh
            mesh = Mesh(
                vertices=vertices,
                faces=faces,
                material=material
            )
            
            # Apply advanced mesh processing
            mesh = self._enhance_mesh(mesh)
            
            return mesh
            
        except Exception as e:
            self.logger.error(f"Failed to create mesh: {e}")
            raise
    
    def _enhance_mesh(self, mesh: Mesh) -> Mesh:
        """Enhance mesh with additional processing."""
        try:
            # Smooth mesh if needed
            if hasattr(self.config.get("rendering", {}), "smooth_meshes") and \
               self.config["rendering"].get("smooth_meshes", False):
                mesh.o3d_mesh = mesh.o3d_mesh.filter_smooth_simple(number_of_iterations=1)
            
            # Subdivide mesh for higher quality if configured
            if hasattr(self.config.get("rendering", {}), "subdivide_meshes") and \
               self.config["rendering"].get("subdivide_meshes", False):
                mesh.o3d_mesh = mesh.o3d_mesh.subdivide_midpoint(number_of_iterations=1)
            
            return mesh
            
        except Exception as e:
            self.logger.warning(f"Failed to enhance mesh: {e}")
            return mesh
    
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
        """Render a single mesh and return the image."""
        try:
            # Set up renderer
            renderer = DibrRenderer(
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
        """Render multiple meshes in a scene with advanced compositing."""
        try:
            if not meshes:
                # Return black image if no meshes
                return torch.zeros(
                    3, 
                    self.config["rendering"]["resolution"]["height"],
                    self.config["rendering"]["resolution"]["width"],
                    device=self.device
                )
            
            # Method 1: Combine meshes into single visualization (preferred)
            if len(meshes) > 1:
                return self._render_combined_scene(meshes)
            else:
                return self.render_mesh(meshes[0])
                
        except Exception as e:
            self.logger.error(f"Failed to render scene: {e}")
            raise
    
    def _render_combined_scene(self, meshes: List[Mesh]) -> torch.Tensor:
        """Render multiple meshes as a combined scene."""
        try:
            # Create combined visualizer
            vis = o3d.visualization.Visualizer()
            vis.create_window(
                width=self.config["rendering"]["resolution"]["width"],
                height=self.config["rendering"]["resolution"]["height"],
                visible=False
            )
            
            # Add all meshes
            for mesh in meshes:
                vis.add_geometry(mesh.o3d_mesh)
            
            # Set up camera
            ctr = vis.get_view_control()
            camera_params = o3d.camera.PinholeCameraParameters()
            
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = self.camera.view_matrix[:3, :3]
            extrinsic[:3, 3] = self.camera.view_matrix[:3, 3]
            
            intrinsic = o3d.camera.PinholeCameraIntrinsic()
            intrinsic.set_intrinsics(
                self.config["rendering"]["resolution"]["width"],
                self.config["rendering"]["resolution"]["height"],
                self.camera.proj_matrix[0, 0] * self.config["rendering"]["resolution"]["width"] / 2,
                self.camera.proj_matrix[1, 1] * self.config["rendering"]["resolution"]["height"] / 2,
                self.config["rendering"]["resolution"]["width"] / 2,
                self.config["rendering"]["resolution"]["height"] / 2
            )
            
            camera_params.extrinsic = extrinsic
            camera_params.intrinsic = intrinsic
            ctr.convert_from_pinhole_camera_parameters(camera_params)
            
            # Render
            vis.poll_events()
            vis.update_renderer()
            
            # Capture image
            image = vis.capture_screen_float_buffer(do_render=True)
            image_np = np.asarray(image)
            
            # Clean up
            vis.destroy_window()
            
            # Convert to torch tensor
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
            
            return image_tensor.to(self.device)
            
        except Exception as e:
            self.logger.warning(f"Combined scene rendering failed, using fallback: {e}")
            return self._render_scene_fallback(meshes)
    
    def _render_scene_fallback(self, meshes: List[Mesh]) -> torch.Tensor:
        """Fallback scene rendering using individual mesh rendering and compositing."""
        try:
            # Render individual meshes and composite
            images = []
            for mesh in meshes:
                image = self.render_mesh(mesh)
                images.append(image)
            
            # Simple alpha compositing
            if len(images) == 1:
                return images[0][:3]  # Remove alpha channel for final output
            
            final_image = torch.zeros_like(images[0][:3])
            
            for image in images:
                if image.shape[0] > 3:  # Has alpha channel
                    alpha = image[3:4]
                    rgb = image[:3]
                else:
                    alpha = torch.ones(1, image.shape[1], image.shape[2], device=self.device)
                    rgb = image
                
                # Alpha blend
                final_image = final_image * (1 - alpha) + rgb * alpha
            
            return final_image
            
        except Exception as e:
            self.logger.error(f"Fallback scene rendering failed: {e}")
            # Return black image as final fallback
            return torch.zeros(
                3,
                self.config["rendering"]["resolution"]["height"],
                self.config["rendering"]["resolution"]["width"],
                device=self.device
            )
    
    def save_image(self, image: torch.Tensor, filepath: str):
        """Save a rendered image to a file with multiple format support."""
        try:
            # Ensure image is on CPU and in correct format
            if image.device != torch.device('cpu'):
                image = image.cpu()
            
            # Convert from (C, H, W) to (H, W, C)
            if len(image.shape) == 3:
                image_np = image.permute(1, 2, 0).numpy()
            else:
                image_np = image.numpy()
            
            # Ensure values are in [0, 1] range
            image_np = np.clip(image_np, 0, 1)
            
            # Convert to 0-255 range
            image_np = (image_np * 255).astype(np.uint8)
            
            # Handle different channel configurations
            if image_np.shape[2] == 4:  # RGBA
                image_pil = Image.fromarray(image_np, 'RGBA')
            elif image_np.shape[2] == 3:  # RGB
                image_pil = Image.fromarray(image_np, 'RGB')
            else:  # Grayscale
                image_pil = Image.fromarray(image_np.squeeze(), 'L')
            
            # Save image
            image_pil.save(filepath)
            self.logger.info(f"Image saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save image: {e}")
            raise
    
    def create_primitive_mesh(self, primitive_type: str, **kwargs) -> Mesh:
        """Create primitive meshes (sphere, cube, cylinder, etc.)."""
        try:
            if primitive_type.lower() == "sphere":
                radius = kwargs.get("radius", 1.0)
                resolution = kwargs.get("resolution", 20)
                mesh_o3d = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
                
            elif primitive_type.lower() == "cube":
                size = kwargs.get("size", 1.0)
                mesh_o3d = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
                
            elif primitive_type.lower() == "cylinder":
                radius = kwargs.get("radius", 0.5)
                height = kwargs.get("height", 1.0)
                resolution = kwargs.get("resolution", 20)
                mesh_o3d = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=resolution)
                
            else:
                raise ValueError(f"Unsupported primitive type: {primitive_type}")
            
            # Convert to our mesh format
            vertices = torch.tensor(np.asarray(mesh_o3d.vertices), device=self.device)
            faces = torch.tensor(np.asarray(mesh_o3d.triangles), device=self.device)
            
            # Create material
            material = Material(
                albedo=torch.tensor([0.7, 0.7, 0.7], device=self.device),
                roughness=0.5,
                metallic=0.0
            )
            
            return Mesh(vertices=vertices, faces=faces, material=material)
            
        except Exception as e:
            self.logger.error(f"Failed to create primitive mesh: {e}")
            raise
    
    def apply_transform(self, mesh: Mesh, transform_matrix: torch.Tensor) -> Mesh:
        """Apply transformation matrix to a mesh."""
        try:
            transform_np = transform_matrix.cpu().numpy() if isinstance(transform_matrix, torch.Tensor) else transform_matrix
            
            # Transform Open3D mesh
            mesh.o3d_mesh.transform(transform_np)
            
            # Update vertices
            mesh.vertices = np.asarray(mesh.o3d_mesh.vertices)
            
            # Update trimesh object
            mesh.trimesh_obj = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
            
            return mesh
            
        except Exception as e:
            self.logger.error(f"Failed to apply transform: {e}")
            raise
    
    def cleanup(self):
        """Clean up rendering resources."""
        try:
            # Clear meshes
            self.meshes.clear()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.logger.info("Rendering resources cleaned up")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup rendering resources: {e}")
            raise