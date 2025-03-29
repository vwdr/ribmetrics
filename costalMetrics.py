import numpy as np
from skimage import measure

def compute_extreme_costal_points(contour_points, center_xy=None, tol=1e-3):
    """
    Given contour points (Nx3), compute the inner and outer costal coordinates based on
    distance from the central vertical axis. Here we assume the vertical axis is fixed and that
    the center (in the horizontal plane) is given by center_xy (or computed as the mean of x and y).
    
    Parameters:
        contour_points (np.ndarray): Nx3 array of points.
        center_xy (tuple or list, optional): (cx, cy). If None, computed as mean of x and y.
        tol (float): Tolerance to treat distances as equal.
    
    Returns:
        inner_point (np.ndarray): The inner costal coordinate.
        outer_point (np.ndarray): The outer costal coordinate.
        inner_distance (float): Radial distance of the inner point from the center.
        outer_distance (float): Radial distance of the outer point from the center.
    """
    if center_xy is None:
        cx = np.mean(contour_points[:, 0])
        cy = np.mean(contour_points[:, 1])
    else:
        cx, cy = center_xy
    
    # Compute radial distances (only x and y; vertical axis assumed to be z)
    r = np.sqrt((contour_points[:, 0] - cx)**2 + (contour_points[:, 1] - cy)**2)
    
    max_r = np.max(r)
    min_r = np.min(r)
    
    # Identify candidates within tolerance
    outer_indices = np.where(np.abs(r - max_r) < tol)[0]
    inner_indices = np.where(np.abs(r - min_r) < tol)[0]
    
    # If multiple candidates, choose the one with median vertical (z) coordinate
    if len(outer_indices) > 1:
        outer_z = contour_points[outer_indices, 2]
        median_index = outer_indices[np.argsort(outer_z)[len(outer_z)//2]]
    else:
        median_index = outer_indices[0]
    outer_point = contour_points[median_index]
    
    if len(inner_indices) > 1:
        inner_z = contour_points[inner_indices, 2]
        median_index = inner_indices[np.argsort(inner_z)[len(inner_z)//2]]
    else:
        median_index = inner_indices[0]
    inner_point = contour_points[median_index]
    
    outer_distance = np.linalg.norm([outer_point[0]-cx, outer_point[1]-cy])
    inner_distance = np.linalg.norm([inner_point[0]-cx, inner_point[1]-cy])
    
    return inner_point, outer_point, inner_distance, outer_distance

def compute_average_rib_width(inner_point, outer_point):
    """    
    Parameters:
        inner_point (array-like): Inner costal coordinate.
        outer_point (array-like): Outer costal coordinate.
    
    Returns:
        float: Rib width.
    """
    return np.linalg.norm(np.array(outer_point) - np.array(inner_point))

def compute_rib_volume(segmentation):

    return int(np.sum(segmentation))

def compute_rib_surface_area(segmentation, level=0.5):
    """
    Parameters:
        segmentation (np.ndarray): 3D binary segmentation array.
        level (float): Threshold for marching cubes.
    
    Returns:
        float: Surface area.
    """
    verts, faces, normals, _ = measure.marching_cubes(segmentation, level=level)

    # Calculate area of each triangle
    triangle_areas = 0.5 * np.linalg.norm(np.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                                                   verts[faces[:, 2]] - verts[faces[:, 0]]), axis=1)
    return np.sum(triangle_areas)

def compute_rib_base_diameter(inner_contour_points):
    """
    Parameters:
        inner_contour_points (np.ndarray): Nx3 array of points along the inner costal contour.
    
    Returns:
        float: Rib base diameter.
    """
    # Efficient pairwise distance computation:
    diff = inner_contour_points[:, np.newaxis, :] - inner_contour_points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    return np.max(dists)
