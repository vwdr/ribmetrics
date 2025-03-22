import numpy as np
from skimage import measure
import trimesh

def get_mesh_from_segmentation(segmentation, level=0.5):
    """
    Compute a surface mesh from a 3D binary segmentation using the marching cubes algorithm.
    
    Parameters:
        segmentation (np.ndarray): A 3D binary array (voxels: 1 = structure, 0 = background).
        level (float): Threshold for marching cubes (default is 0.5 for binary images).
    
    Returns:
        trimesh.Trimesh: The computed 3D surface mesh.
    """
    verts, faces, normals, _ = measure.marching_cubes(segmentation, level=level)
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
