import numpy as np
import trimesh
from mesh import get_mesh_from_segmentation

class ObliquePlane:
    """
    Represents an oblique cross-sectional plane in 3D space.
    The plane is defined by a normal vector and a point on the plane.
    """
    def __init__(self, normal, point):
        """
        Parameters:
            normal (array-like of 3 floats): The normal vector to the plane.
            point  (array-like of 3 floats): A point through which the plane passes.
        """
        normal = np.array(normal, dtype=float)
        point = np.array(point, dtype=float)
        if normal.shape != (3,) or point.shape != (3,):
            raise ValueError("Normal vector and point must be 3-element sequences.")
        norm = np.linalg.norm(normal)
        if norm == 0:
            raise ValueError("Normal vector must not be zero.")
        self.normal = normal / norm
        self.point = point

    def point_distance(self, pt):
        """
        Calculate the signed distance from a point to the plane.
        
        Parameters:
            pt (array-like): 3D point.
            
        Returns:
            float: Signed distance from the point to the plane.
        """
        pt = np.array(pt, dtype=float)
        return np.dot(self.normal, pt - self.point)

    def project_point(self, pt):
        """
        Project a 3D point onto the plane.
        
        Parameters:
            pt (array-like): 3D point.
            
        Returns:
            np.ndarray: The projected point on the plane.
        """
        pt = np.array(pt, dtype=float)
        d = self.point_distance(pt)
        return pt - d * self.normal

    def get_cross_section(self, segmentation, level=0.5):
        """
        Compute the cross-sectional contour(s) by intersecting the surface mesh of a segmentation
        with this plane.
        
        Parameters:
            segmentation (np.ndarray): A 3D binary segmentation array.
            level (float): Threshold for mesh extraction (default is 0.5).
        
        Returns:
            trimesh.path.Path3D or None: The intersection contour(s) as a 3D path, or None if no intersection is found.
        """
        mesh = get_mesh_from_segmentation(segmentation, level=level)
        section = mesh.section(plane_origin=self.point, plane_normal=self.normal)
        return section

    def __repr__(self):
        a, b, c = self.normal
        d = -np.dot(self.normal, self.point)
        return (f"ObliquePlane(normal={self.normal.tolist()}, point={self.point.tolist()}, "
                f"equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0)")
