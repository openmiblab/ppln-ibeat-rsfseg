import logging

import numpy as np
from skimage import measure
from scipy.spatial import ConvexHull, distance_matrix


def shape_features_3d(input_mask, voxel_spacing):
    r"""
    Returns the shape features of the pyradiomics package
    """

    assert (
        input_mask.ndim == 3
    ), "Shape features are only available in 3D. "
    
    # Zero-pad mask
    mask = np.pad(input_mask, pad_width=3, mode='constant', constant_values=0)
    
    # Extract surface mesh using marching cubes
    verts, faces, normals, _ = measure.marching_cubes(mask, spacing=voxel_spacing)
    
    # Compute Surface Area
    def mesh_surface_area(vertices, faces):
        area = 0.0
        for tri in faces:
            v0, v1, v2 = vertices[tri]
            # Triangle area using cross product formula
            area += 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        return area
    
    surface_area = mesh_surface_area(verts, faces)
    
    # Compute Mesh Volume using divergence theorem
    def mesh_volume(vertices, faces):
        vol = 0.0
        for tri in faces:
            v0, v1, v2 = vertices[tri]
            vol += np.dot(v0, np.cross(v1, v2)) / 6.0
        return abs(vol)
    
    volume = mesh_volume(verts, faces)
    
    # Maximum 3D diameter
    def max_diameter(points):
        hull = ConvexHull(points)
        hull_pts = points[hull.vertices]
        dist_matrix = distance_matrix(hull_pts, hull_pts)
        return np.max(dist_matrix)
    
    # Maximum 3D diameter
    max_3D_diameter = max_diameter(verts)
    
    # Maximum 2D diameters
    max_2D_slice = max_diameter(verts[:, [0,1]])    # row-column
    max_2D_column = max_diameter(verts[:, [0,2]])   # row-slice
    max_2D_row = max_diameter(verts[:, [1,2]])      # column-slice

    # Eigen decomposition of covariance of vertices
    cov_matrix = np.cov(verts.T)
    eigenvalues, _ = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues/eigenvectors in ascending order
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]

    # Compute derived parameters
    voxel_volume = np.sum(input_mask != 0) * np.prod(voxel_spacing)
    surface_volume_ratio = surface_area / volume
    sphericity = (36 * np.pi * volume**2) ** (1.0 / 3.0) / surface_area
    major_axis_length = np.nan if eigenvalues[2] < 0 else np.sqrt(eigenvalues[2]) * 4
    minor_axis_length = np.nan if eigenvalues[1] < 0 else np.sqrt(eigenvalues[1]) * 4
    least_axis_length = np.nan if eigenvalues[0] < 0 else np.sqrt(eigenvalues[0]) * 4
    elongation = np.nan if (eigenvalues[1] < 0 or eigenvalues[2] < 0) else np.sqrt(eigenvalues[1] / eigenvalues[2])
    flatness = np.nan if (eigenvalues[0] < 0 or eigenvalues[2] < 0) else np.sqrt(eigenvalues[0] / eigenvalues[2])

    # Issue warnings
    if least_axis_length == np.nan:
        logging.warning(
            "Least axis eigenvalue negative! (%g)", eigenvalues[0]
        )
    if minor_axis_length == np.nan:
        logging.warning(
            "Minor axis eigenvalue negative! (%g)", eigenvalues[1]
        )
    if major_axis_length == np.nan:
        logging.warning(
            "Major axis eigenvalue negative! (%g)", eigenvalues[2]
        )
    if elongation == np.nan:
        logging.warning(
            "Elongation eigenvalue negative! (%g, %g)",
            eigenvalues[1],
            eigenvalues[2],
        )
    if flatness == np.nan:
        logging.warning(
            "Flatness eigenvalue negative! (%g, %g)",
            eigenvalues[0],
            eigenvalues[2],
        )

    # Build return values
    return {
        "SurfaceArea": surface_area,
        "MeshVolume": volume,
        "Maximum2DDiameterSlice": max_2D_slice,
        "Maximum2DDiameterColumn": max_2D_column,
        "Maximum2DDiameterRow": max_2D_row,
        "Maximum3DDiameter": max_3D_diameter,
        "VoxelVolume": voxel_volume,
        "SurfaceVolumeRatio": surface_volume_ratio,
        "Sphericity": sphericity,
        "MajorAxisLength": major_axis_length,
        "MinorAxisLength": minor_axis_length,
        "LeastAxisLength": least_axis_length,
        "Elongation": elongation,
        "Flatness": flatness,
    }
    
