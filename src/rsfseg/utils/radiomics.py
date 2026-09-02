import os
import logging
import tempfile

import numpy as np
import scipy.ndimage as ndi
from scipy.interpolate import interpn
import skimage
# import vreg
# from radiomics import featureextractor

from rsfseg.utils import numpyradiomics as nprad


biomarker_units = {
    "shape_Elongation": "%",                 # ratio of axes
    "shape_Flatness": "%",                   # ratio of axes
    "shape_LeastAxisLength": "cm",                       # shortest principal axis
    "shape_MajorAxisLength": "cm",                       # longest principal axis
    "shape_Maximum2DDiameterColumn": "cm",               # max 2D distance in column direction
    "shape_Maximum2DDiameterRow": "cm",                  # max 2D distance in row direction
    "shape_Maximum2DDiameterSlice": "cm",                # max 2D distance in slice direction
    "shape_Maximum3DDiameter": "cm",                     # max 3D distance
    "shape_MeshVolume": "mL",                          # volume of mesh
    "shape_MinorAxisLength": "cm",                       # intermediate principal axis
    "shape_Sphericity": "%",                 # ratio (1 = perfect sphere)
    "shape_SurfaceArea": "cm^2",                         # area of the surface
    "shape_SurfaceVolumeRatio": "cm^-1",                 # SurfaceArea / MeshVolume
    "shape_VoxelVolume": "mL",                         # voxel count × voxel spacing
}

conversion_factor = {
    "shape_Elongation": 100,                 # ratio of axes
    "shape_Flatness": 100,                   # ratio of axes
    "shape_LeastAxisLength": 1/10,                       # shortest principal axis
    "shape_MajorAxisLength": 1/10,                       # longest principal axis
    "shape_Maximum2DDiameterColumn": 1/10,               # max 2D distance in column direction
    "shape_Maximum2DDiameterRow": 1/10,                  # max 2D distance in row direction
    "shape_Maximum2DDiameterSlice": 1/10,                # max 2D distance in slice direction
    "shape_Maximum3DDiameter": 1/10,                     # max 3D distance
    "shape_MeshVolume": 1/1000,                          # volume of mesh
    "shape_MinorAxisLength": 1/10,                       # intermediate principal axis
    "shape_Sphericity": 100,                 # ratio (1 = perfect sphere) -> should be the same as compactness
    "shape_SurfaceArea": 1/100,                         # area of the surface
    "shape_SurfaceVolumeRatio": 10,                 # SurfaceArea / MeshVolume
    "shape_VoxelVolume": 1/1000,                         # voxel count × voxel spacing
}


def interpolate3d_isotropic(array, spacing, isotropic_spacing=None):

    if isotropic_spacing is None:
        isotropic_spacing = np.amin(spacing)

    # Get x, y, z coordinates for array
    nx = array.shape[0]
    ny = array.shape[1]
    nz = array.shape[2]
    Lx = (nx-1)*spacing[0]
    Ly = (ny-1)*spacing[1]
    Lz = (nz-1)*spacing[2]
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    z = np.linspace(0, Lz, nz)

    # Get x, y, z coordinates for isotropic array
    nxi = 1 + np.floor(Lx/isotropic_spacing)
    nyi = 1 + np.floor(Ly/isotropic_spacing)
    nzi = 1 + np.floor(Lz/isotropic_spacing)
    Lxi = (nxi-1)*isotropic_spacing
    Lyi = (nyi-1)*isotropic_spacing
    Lzi = (nzi-1)*isotropic_spacing
    xi = np.linspace(0, Lxi, nxi.astype(int))
    yi = np.linspace(0, Lyi, nyi.astype(int))
    zi = np.linspace(0, Lzi, nzi.astype(int))

    # Interpolate to isotropic
    ri = np.meshgrid(xi,yi,zi, indexing='ij')
    array = interpn((x,y,z), array, np.stack(ri, axis=-1))
    return array, isotropic_spacing


def volume_features(vol, roi):

    arr = vol.values

    # Add zeropadding at the boundary slices for masks that extend to the edge
    # Motivation: this could have some effect if surfaces are extracted - could create issues
    # if the values extend right up to the boundary of the slab.
    shape = list(arr.shape)
    shape[-1] = shape[-1] + 2*4
    array = np.zeros(shape)
    array[:,:,4:-4] = arr

    # Get voxel dimensions from the affine
    # We are assuming here the voxel dimensions are in mm.
    # If not the units provided with the return values are incorrect.
    spacing = vol.spacing
    voxel_volume = spacing[0]*spacing[1]*spacing[2]
    nr_of_voxels = np.count_nonzero(array > 0.5)
    volume = nr_of_voxels * voxel_volume

    # Surface properties - for now only extracting surface area
    try:
        # Note: this is smoothing the surface first - not tested in depth whether this is necessary or helpful.
        # It does appear to make a big difference on surface area so should be looked at more carefully.
        smooth_array = ndi.gaussian_filter(array, 1.0)
        verts, faces, _, _ = skimage.measure.marching_cubes(smooth_array, spacing=spacing, level=0.5, step_size=1.0)
    except:
        # If a mask has too few points, smoothing can reduce the max to below 0.5. Use the midpoint in that case
        # Note this may work in general but 0.5 has been used for previous data collection so keep that as default
        smooth_array = ndi.gaussian_filter(array, 1.0)
        verts, faces, _, _ = skimage.measure.marching_cubes(smooth_array, spacing=spacing, level=np.mean(smooth_array), step_size=1.0)
    surface_area = skimage.measure.mesh_surface_area(verts, faces)

    # Interpolate to isotropic for non-isotropic voxels
    # Motivation: this is required by the region_props function
    spacing = np.array(spacing)
    if np.amin(spacing) != np.amax(spacing):
        array, isotropic_spacing = interpolate3d_isotropic(array, spacing)
        isotropic_voxel_volume = isotropic_spacing**3
    else:
        isotropic_spacing = np.mean(spacing)
        isotropic_voxel_volume = voxel_volume

    # Get volume properties - mostly from region_props, except for compactness and depth
    array = np.round(array).astype(np.int16)
    region_props_3D = skimage.measure.regionprops(array)[0]

    # Calculate 'compactness' (our definition) - define as volume to surface ratio
    # expressed as a percentage of the volume-to-surface ration of an equivalent sphere.
    # The sphere is the most compact of all shapes, i.e. it has the largest volume to surface area ratio,
    # so this is guaranteed to be between 0 and 100%
    radius = region_props_3D['equivalent_diameter_area']*isotropic_spacing/2 # mm
    v2s = volume/surface_area # mm
    v2s_equivalent_sphere = radius/3 # mm
    compactness = 100 * v2s/v2s_equivalent_sphere # %

    # Fractional anisotropy - in analogy with FA in diffusion 
    m0 = region_props_3D['inertia_tensor_eigvals'][0]
    m1 = region_props_3D['inertia_tensor_eigvals'][1]
    m2 = region_props_3D['inertia_tensor_eigvals'][2]

    # # Pull the raw 3x3 inertia matrix
    # inertia_matrix = region_props_3D.inertia_tensor
    
    # # Compute eigenvalues explicitly using numpy
    # eigvals = np.linalg.eigvals(inertia_matrix)
    
    # # Sort them in descending order to match skimage behavior
    # eigvals = np.sort(eigvals)[::-1]

    # m0, m1, m2 = eigvals[0], eigvals[1], eigvals[2]

    m = (m0 + m1 + m2)/3 # average moment of inertia (trace of the inertia tensor)
    FA = np.sqrt(3/2) * np.sqrt((m0-m)**2 + (m1-m)**2 + (m2-m)**2) / np.sqrt(m0**2 + m1**2 + m2**2)

    # Measure maximum depth (our definition)
    distance = ndi.distance_transform_edt(array)
    max_depth = np.amax(distance)

    # Adding a try/except around each line as some of these fail (math error) for masks with limited non-zero values
    data = {}
    try:
        data[f'{roi}-shape_ski-surface_area'] = [surface_area/100, f'{roi} surface area', 'cm^2', 'float']
    except Exception as e:
        logging.error(f"Error computing surface area ({roi}): {e}")
    try:
        data[f'{roi}-shape_ski-volume'] = [volume/1000, f'{roi} volume', 'mL', 'float']
    except Exception as e:
        logging.error(f"Error computing Volume ({roi}): {e}")
    try:
        data[f'{roi}-shape_ski-bounding_box_volume'] = [region_props_3D['area_bbox']*isotropic_voxel_volume/1000, f'{roi} bounding box volume', 'mL', 'float']
    except Exception as e:
        logging.error(f"Error computing Bounding box volume ({roi}): {e}")
    try:
        data[f'{roi}-shape_ski-convex_hull_volume'] = [region_props_3D['area_convex']*isotropic_voxel_volume/1000, f'{roi} convex hull volume', 'mL', 'float']
    except Exception as e:
        logging.error(f"Error computing Convex hull volume ({roi}): {e}")
    # try:
    #     data[f'{roi}-shape_ski-volume_of_holes'] = [(region_props_3D['area_filled']-region_props_3D['area'])*isotropic_voxel_volume/1000, f'Volume of holes ({roi})', 'mL', 'float']
    # except Exception as e:
    #     logging.error(f"Error computing Volume of holes ({roi}): {e}")
    try:
        data[f'{roi}-shape_ski-extent'] = [region_props_3D['extent']*100, f'{roi} extent', '%', 'float']    # Percentage of bounding box filled
    except Exception as e:
        logging.error(f"Error computing Extent ({roi}): {e}")
    try:
        data[f'{roi}-shape_ski-solidity'] = [region_props_3D['solidity']*100, f'{roi} solidity', '%', 'float']   # Percentage of convex hull filled
    except Exception as e:
        logging.error(f"Error computing Solidity ({roi}): {e}")
    try:
        data[f'{roi}-shape_ski-compactness'] = [compactness, f'{roi} compactness', '%', 'float']
    except Exception as e:
        logging.error(f"Error computing Compactness ({roi}): {e}")
    try:
        data[f'{roi}-shape_ski-long_axis_length'] = [region_props_3D['axis_major_length']*isotropic_spacing/10, f'{roi} long axis length', 'cm', 'float']
    except Exception as e:
        logging.error(f"Error computing Long axis length ({roi}): {e}")
    try:
        data[f'{roi}-shape_ski-short_axis_length'] = [region_props_3D['axis_minor_length']*isotropic_spacing/10, f'{roi} short axis length', 'cm', 'float']
    except Exception as e:
        logging.error(f"Error computing Short axis length ({roi}): {e}")
    try:
        data[f'{roi}-shape_ski-equivalent_diameter'] = [region_props_3D['equivalent_diameter_area']*isotropic_spacing/10, f'{roi} equivalent diameter', 'cm', 'float']
    except Exception as e:
        logging.error(f"Error computing Equivalent diameter ({roi}): {e}")
    try:
        data[f'{roi}-shape_ski-maximum_depth'] = [max_depth*isotropic_spacing/10, f'{roi} maximum depth', 'cm', 'float']
    except Exception as e:
        logging.error(f"Error computing Maximum depth ({roi}): {e}")
    try:
        data[f'{roi}-shape_ski-primary_moment_of_inertia'] = [region_props_3D['inertia_tensor_eigvals'][0]*isotropic_spacing**2/100, f'{roi} primary moment of inertia', 'cm^2', 'float']
    except Exception as e:
        logging.error(f"Error computing Primary moment of inertia ({roi}): {e}")
    try:
        data[f'{roi}-shape_ski-second_moment_of_inertia'] = [region_props_3D['inertia_tensor_eigvals'][1]*isotropic_spacing**2/100, f'{roi} second moment of inertia', 'cm^2', 'float']
    except Exception as e:
        logging.error(f"Error computing Second moment of inertia ({roi}): {e}")
    try:
        data[f'{roi}-shape_ski-third_moment_of_inertia'] = [region_props_3D['inertia_tensor_eigvals'][2]*isotropic_spacing**2/100, f'{roi} third moment of inertia', 'cm^2', 'float']
    except Exception as e:
        logging.error(f"Error computing Third moment of inertia ({roi}): {e}")
    try:
        data[f'{roi}-shape_ski-mean_moment_of_inertia'] = [m*isotropic_spacing**2/100, f'{roi} mean moment of inertia', 'cm^2', 'float']
    except Exception as e:
        logging.error(f"Error computing Mean moment of inertia ({roi}): {e}")
    try:
        data[f'{roi}-shape_ski-fractional_anisotropy_of_inertia'] = [100*FA, f'{roi} fractional anisotropy of inertia', '%', 'float']
    except Exception as e:
        logging.error(f"Error computing Fractional anisotropy of inertia ({roi}): {e}")
    try:
        data[f'{roi}-shape_ski-volume_qc'] = [region_props_3D['area']*isotropic_voxel_volume/1000, f'{roi} volume QC', 'mL', 'float']
    except Exception as e:
        logging.error(f"Error computing Volume QC ({roi}): {e}")
    # Taking this out for now - computation uses > 32GB of memory for large masks
    # data[f'{roi}_ski_longest_caliper_diameter'] = [region_props_3D['feret_diameter_max']*isotropic_spacing/10, f'Longest caliper diameter ({roi})', 'cm', 'float']

    return data

# pyradiomics taken out for now because of installation issues
# 
# def shape_features(roi_vol, roi):

#     with tempfile.TemporaryDirectory() as tmp:
#         roi_file = os.path.join(tmp, 'roi.nii.gz')
#         img_file = os.path.join(tmp, 'img.nii.gz') # dummy
#         vreg.write_nifti(roi_vol, roi_file)
#         vreg.write_nifti(roi_vol, img_file)
#         # All features for water
#         extractor = featureextractor.RadiomicsFeatureExtractor()
#         extractor.disableAllFeatures()
#         extractor.enableFeatureClassByName('shape')
#         # Enable only the shape features you care about
#         extractor.enableFeaturesByName(
#             shape = [
#                 "Elongation",
#                 "Flatness",
#                 "LeastAxisLength", # seems unstable
#                 "MajorAxisLength",
#                 "Maximum2DDiameterColumn",
#                 "Maximum2DDiameterRow",
#                 "Maximum2DDiameterSlice",
#                 "Maximum3DDiameter",
#                 "MeshVolume",
#                 "MinorAxisLength",
#                 "Sphericity",
#                 "SurfaceArea",
#                 "SurfaceVolumeRatio",
#                 "VoxelVolume",
#             ]
#         )
#         result = extractor.execute(img_file, roi_file)
        
#     # Format return value
#     rval = {}
#     for p, v in result.items():
#         if p[:8]=='original':
#             name = roi + '-' + p.replace('original_shape_', 'shape_rad-')
#             desc = f"{roi} {p.replace('original_shape_', '')}"
#             unit = biomarker_units[p.replace('original_shape_', 'shape_')]
#             v = float(v) * conversion_factor[p.replace('original_shape_', 'shape_')]
#             vals = [v, desc, unit, 'float']
#             rval[name] = vals
#     return rval


def shape_features_nprad(roi_vol, roi):

    result = nprad.shape_features_3d(roi_vol.values, roi_vol.spacing)

    rval = {}
    for p, v in result.items():
        name = roi + '-shape_nprad-' + p
        desc = f"{roi} {p}"
        unit = biomarker_units['shape_' + p]
        v = float(v) * conversion_factor['shape_' + p]
        vals = [v, desc, unit, 'float']
        rval[name] = vals
    return rval


