import numpy as np
from ObliquePlane import ObliquePlane
from costalMetrics import (compute_extreme_costal_points, compute_average_rib_width,
                            compute_rib_volume, compute_rib_surface_area, compute_rib_base_diameter)
from mesh import get_mesh_from_segmentation
import trimesh
import matplotlib.pyplot as plt

'''
Note: just set segmentation = to your Binary 3D array and run
all objects should be made
'''

if __name__ == "__main__":
    
    #oblique plane 
    plane = ObliquePlane(normal=[1, 1, 0], point=[32, 32, 32])
    
    #cross-sectional contour
    section = plane.get_cross_section(segmentation) #parameter is segmentation
    if section is None:
        print("No cross section found.")
    else:
        # Combine all vertices from the cross-sectional entities.
        contour_points = []
        for entity in section.entities:
            contour_points.extend(entity.points)
        contour_points = np.array(contour_points)
        
        # inner and outer costal coordinates from the contour.
        inner_point, outer_point, inner_dist, outer_dist = compute_extreme_costal_points(contour_points)
        print("Inner costal coordinate:", inner_point)
        print("Outer costal coordinate:", outer_point)
        print("Inner costal distance:", inner_dist)
        print("Outer costal distance:", outer_dist)
        
        # approx rib width as the dist. btwen the extreme costal coordinates.
        rib_width = compute_average_rib_width(inner_point, outer_point)
        print("Approximate rib width:", rib_width)
        
        # get the inner costal contour.
        cx = np.mean(contour_points[:, 0])
        cy = np.mean(contour_points[:, 1])
        r = np.sqrt((contour_points[:, 0] - cx)**2 + (contour_points[:, 1] - cy)**2)
        median_r = np.median(r)
        inner_contour_points = contour_points[r < median_r]
        if inner_contour_points.size > 0:
            rib_base_diam = compute_rib_base_diameter(inner_contour_points)
            print("Rib base diameter:", rib_base_diam)
        else:
            print("Insufficient inner contour points for rib base diameter.")
        
        # rib volume
        rib_vol = compute_rib_volume(segmentation)
        print("Rib volume (voxel count):", rib_vol)
        
        # rib surface area.
        rib_area = compute_rib_surface_area(segmentation)
        print("Rib surface area:", rib_area)
        
        #visualize the cross-sectional contour using trimesh's viewer.
        section.show()
