import pyvista as pv
import numpy as np

def surface_with_clipping(
        masks, spacing=(1.0,1.0,1.0),
        colors=None, opacity=1.0,
        above_opacity=0.1,
        n_iter=30, relaxation=0.1):
    """
    Display multiple surfaces with an interactive clipping plane:
    - Below the plane: colored (cross-section)
    - Above the plane: semi-transparent gray
    - Plane itself: neutral wireframe (gray)
    """
    if isinstance(masks, np.ndarray):
        masks = [masks]

    n_masks = len(masks)
    if colors is None:
        default_colors = ["lightblue", "lightcoral", "lightgreen", "wheat", "thistle", "rotalblue", "goldenrod"]
        colors = [default_colors[i % len(default_colors)] for i in range(n_masks)]

    if isinstance(opacity, (float,int)):
        opacities = [opacity]*n_masks
    else:
        opacities = opacity

    plotter = pv.Plotter()

    original_meshes = []
    actors_below = []
    actors_above = []

    for mask, color, opac in zip(masks, colors, opacities):
        grid = pv.wrap(mask.astype(np.uint8))
        grid.spacing = spacing
        contour = grid.contour(isosurfaces=[0.5])
        if n_iter>0:
            contour = contour.smooth(n_iter=n_iter, relaxation_factor=relaxation)

        # Below-plane: colored
        actor_below = plotter.add_mesh(contour.copy(), color=color,
                                       smooth_shading=True, opacity=opac)
        # Above-plane: grey, semi-transparent
        actor_above = plotter.add_mesh(contour.copy(), color='gray',
                                       smooth_shading=True, opacity=above_opacity)

        original_meshes.append(contour)
        actors_below.append(actor_below)
        actors_above.append(actor_above)

    # Optional faint context
    base_grid = pv.wrap(masks[0].astype(np.uint8))
    base_grid.spacing = spacing
    context_surface = base_grid.contour(isosurfaces=[0.5])
    plotter.add_mesh(context_surface, color='gray', opacity=0.05, smooth_shading=False)

    # Callback for interactive clipping
    def plane_callback(normal, origin):
        for orig, below, above in zip(original_meshes, actors_below, actors_above):
            # Below plane
            clipped_below = orig.clip(normal=normal, origin=origin, invert=False)
            below.mapper.SetInputData(clipped_below)
            # Above plane (grey, transparent)
            clipped_above = orig.clip(normal=normal, origin=origin, invert=True)
            above.mapper.SetInputData(clipped_above)

    # Add interactive plane widget (neutral gray wireframe)
    plotter.add_plane_widget(
        callback=plane_callback,
        normal='z',
        assign_to_axis='z',
        color='#404040',             # neutral color
        outline_translation=True   # wireframe-like outline
    )

    # Lighting
    plotter.add_light(pv.Light(position=(1,1,1), focal_point=(0,0,0), intensity=0.4))
    plotter.add_light(pv.Light(position=(-1,-1,1), focal_point=(0,0,0), intensity=0.2))

    plotter.show()
    return actors_below, actors_above






def surface(masks, spacing=(1.0, 1.0, 1.0),
            colors=None, opacity=1.0,
            n_iter=30, relaxation=0.1):
    """
    Extract and display one or more surface meshes from binary masks,
    correcting for anisotropic voxels (shared spacing).

    Parameters
    ----------
    masks : list of np.ndarray or np.ndarray
        One or more 3D binary masks (x, y, z).
    spacing : tuple of float
        Voxel spacing in (x, y, z) order (applies to all masks).
    colors : list of str/tuple, optional
        Colors for each mask. Defaults to a small built-in palette if None.
    opacity : float or list of float
        Either a single opacity value for all masks,
        or a list of values (0.0 transparent, 1.0 opaque) per mask.
    n_iter : int
        Number of smoothing iterations (default=30).
    relaxation : float
        Relaxation factor for smoothing (default=0.1).
    """
    # Ensure list input
    if isinstance(masks, np.ndarray):
        masks = [masks]

    n_masks = len(masks)

    # Handle defaults
    if colors is None:
        default_colors = ["lightblue", "lightcoral", "lightgreen", "wheat", "thistle", "rotalblue", "goldenrod"]
        colors = [default_colors[i % len(default_colors)] for i in range(n_masks)]

    # Normalize opacity input
    if isinstance(opacity, (float, int)):
        opacities = [opacity] * n_masks
    else:
        opacities = opacity

    # Create plotter
    plotter = pv.Plotter()
    meshes = []

    for mask, color, opac in zip(masks, colors, opacities):
        # Wrap mask into PyVista ImageData
        grid = pv.wrap(mask.astype(np.uint8))
        grid.spacing = spacing

        # Extract iso-surface
        contour = grid.contour(isosurfaces=[0.5])

        # Optional smoothing
        if n_iter > 0:
            contour = contour.smooth(n_iter=n_iter, relaxation_factor=relaxation)

        # Add to scene
        plotter.add_mesh(
            contour,
            color=color,
            smooth_shading=True,
            opacity=opac
        )
        meshes.append(contour)

    # Add light for depth effect
    plotter.add_light(pv.Light(position=(1, 1, 1), focal_point=(0, 0, 0),
                               color='white', intensity=0.4))

    # Show combined scene
    plotter.show()

    return meshes












def display_both_surfaces(mask, mask_recon):
    # ---------------------------
    # 7. Visualize with PyVista
    # ---------------------------
    # Original mesh
    grid_orig = pv.wrap(mask.astype(np.uint8))
    contour_orig = grid_orig.contour(isosurfaces=[0.5])

    # Reconstructed mesh
    grid_recon = pv.wrap(mask_recon.astype(np.uint8))
    contour_recon = grid_recon.contour(isosurfaces=[0.5])

    plotter = pv.Plotter(shape=(1,2))
    plotter.subplot(0,0)
    plotter.add_text("Original", font_size=12)
    plotter.add_mesh(contour_orig, color="lightblue")

    plotter.subplot(0,1)
    plotter.add_text("Reconstructed", font_size=12)
    plotter.add_mesh(contour_recon, color="salmon")

    plotter.show()


def display_volumes(original_volume, reconstructed_volume):
    """
    Uses pyvista to display the original and reconstructed volumes.

    The original volume is shown with a transparent surface, while the
    reconstructed volume is shown as a solid surface.
    """
    plotter = pv.Plotter(window_size=(800, 600))
    plotter.background_color = 'white'
    
    # Create the original volume mesh from the numpy array
    original_mesh = pv.wrap(original_volume)
    original_surface = original_mesh.contour(isosurfaces=[0.5])
    
    # Add the original surface to the plotter with transparency
    plotter.add_mesh(
        original_surface, 
        color='blue', 
        opacity=0.3, 
        style='surface', 
        label='Original Volume'
    )
    
    # Create the reconstructed volume mesh
    reconstructed_mesh = pv.wrap(reconstructed_volume)
    reconstructed_surface = reconstructed_mesh.contour(isosurfaces=[0.1])
    
    # Add the reconstructed surface to the plotter as a solid mesh
    plotter.add_mesh(
        reconstructed_surface, 
        color='red', 
        style='surface', 
        label=f'Reconstructed Volume'
    )
    
    plotter.add_legend()
    plotter.add_text(f'3D Volume Reconstruction', font_size=20)
    
    # Set the camera position and show the plot
    plotter.camera_position = 'iso'
    plotter.show()