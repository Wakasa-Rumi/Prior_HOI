import torch
import numpy as np
from sdf import SDFLoss
import torch.nn.functional as torch_f
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer, 
    SoftSilhouetteShader
)
from pytorch3d.structures.meshes import Meshes

class Render:
    def __init__(self, angle1=0, angle2=0):
        R, T = look_at_view_transform(2.7, angle1, angle2)
        self.cameras = FoVPerspectiveCameras(device=torch.device('cuda'), R=R, T=T)

        # Rasterization settings for silhouette rendering  
        sigma = 1e-4
        raster_settings_silhouette = RasterizationSettings(
            image_size=128, 
            blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
            faces_per_pixel=50, 
        )
        # Silhouette renderer 
        self.renderer_silhouette = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings_silhouette
            ),
            shader=SoftSilhouetteShader()
        )
    
    def rend(self, mesh):
        silhouette_images = self.renderer_silhouette(mesh, cameras=self.cameras)
        s = silhouette_images[:,:,:,3]
        return s