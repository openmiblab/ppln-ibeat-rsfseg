import os
import shutil
import logging

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
# import imageio.v2 as imageio  # Use v2 interface for compatibility
# from moviepy import VideoFileClip




def get_distinct_colors(rois, colormap='jet'):
    if len(rois)==1:
        colors = [[255, 0, 0, 0.6]]
    elif len(rois)==2:
        colors = [[255, 0, 0, 0.6], [0, 255, 0, 0.6]]
    elif len(rois)==3:
        colors = [[255, 0, 0, 0.6], [0, 255, 0, 0.6], [0, 0, 255, 0.6]]
    else:
        n = len(rois)
        #cmap = cm.get_cmap(colormap, n)
        cmap = matplotlib.colormaps[colormap]
        colors = [cmap(i)[:3] + (0.6,) for i in np.linspace(0, 1, n)]  # Set alpha to 0.6 for transparency

    return colors


# def movie_overlay(img, rois, file):

#     # Define RGBA colors (R, G, B, Alpha) — alpha controls transparency
#     colors = get_distinct_colors(rois, colormap='tab20')

#     # Directory to store temporary frames
#     tmp = os.path.join(os.getcwd(), 'tmp')
#     os.makedirs(tmp, exist_ok=True)
#     filenames = []

#     # Generate and save a sequence of plots
#     for i in tqdm(range(img.shape[2]), desc='Building animation..'):

#         # Set up figure
#         fig, ax = plt.subplots(
#             figsize=(5, 5),
#             dpi=300,
#         )

#         # Display the background image
#         ax.imshow(img[:,:,i].T, cmap='gray', interpolation='none', vmin=0, vmax=np.mean(img) + 2 * np.std(img))

#         # Overlay each mask
#         for mask, color in zip([m.astype(bool) for m in rois.values()], colors):
#             rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=float)
#             for c in range(4):  # RGBA
#                 rgba[..., c] = mask[:,:,i] * color[c]
#             ax.imshow(rgba.transpose((1,0,2)), interpolation='none')

#         # Save eachg image to a tmp file
#         fname = os.path.join(tmp, f'frame_{i}.png')
#         fig.savefig(fname)
#         filenames.append(fname)
#         plt.close(fig)

#     # Create GIF
#     print('Creating movie')
#     gif = os.path.join(tmp, 'movie.gif')
#     with imageio.get_writer(gif, mode="I", duration=0.2) as writer:
#         for fname in filenames:
#             image = imageio.imread(fname)
#             writer.append_data(image)

#     # Load gif
#     clip = VideoFileClip(gif)

#     # Save as MP4
#     clip.write_videofile(file, codec='libx264')

#     # Clean up temporary files
#     shutil.rmtree(tmp)


def mosaic_overlay(img, rois, file, colormap='tab20', aspect_ratio=16/9, margin=[15,5,2]):

    # Define RGBA colors (R, G, B, Alpha) — alpha controls transparency
    colors = get_distinct_colors(rois, colormap=colormap)

    # Get all masks as boolean arrays
    masks = [m.astype(bool) for m in rois.values()]

    # Build a single combined mask
    all_masks = masks[0]
    for i in range(1, len(masks)):
        all_masks = np.logical_or(all_masks, masks[i])
    if np.sum(all_masks)==0:
        logging.error(f'Empty masks, no plot saved to {file}')
        return
    
    # Find corners of cropped mask
    for x0 in range(all_masks.shape[0]):
        if np.sum(all_masks[x0,:,:]) > 0:
            break
    for x1 in range(all_masks.shape[0]-1, -1, -1):
        if np.sum(all_masks[x1,:,:]) > 0:
            break
    for y0 in range(all_masks.shape[1]):
        if np.sum(all_masks[:,y0,:]) > 0:
            break
    for y1 in range(all_masks.shape[1]-1, -1, -1):
        if np.sum(all_masks[:,y1,:]) > 0:
            break
    for z0 in range(all_masks.shape[2]):
        if np.sum(all_masks[:,:,z0]) > 0:
            break
    for z1 in range(all_masks.shape[2]-1, -1, -1):
        if np.sum(all_masks[:,:,z1]) > 0:
            break

    # Add in the margins       
    x0 = x0-margin[0] if x0-margin[0]>=0 else 0
    y0 = y0-margin[1] if y0-margin[1]>=0 else 0
    z0 = z0-margin[2] if z0-margin[2]>=0 else 0
    x1 = x1+margin[0] if x1+margin[0]<all_masks.shape[0] else all_masks.shape[0]-1
    y1 = y1+margin[1] if y1+margin[1]<all_masks.shape[1] else all_masks.shape[1]-1
    z1 = z1+margin[2] if z1+margin[2]<all_masks.shape[2] else all_masks.shape[2]-1

    # Determine number of rows and columns
    # c*r = n -> c=n/r
    # c*w / r*h = a -> w*n/r = a*r*h -> (w*n) / (a*h) = r**2
    width = x1-x0+1
    height = y1-y0+1
    n_mosaics = z1-z0+1
    nrows = int(np.round(np.sqrt((width*n_mosaics)/(aspect_ratio*height))))
    ncols = int(np.ceil(n_mosaics/nrows))

    # Set up figure 
    fig, ax = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        gridspec_kw = {'wspace':0, 'hspace':0}, 
        figsize=(ncols*width/max([width,height]), nrows*height/max([width,height])),
        dpi=300,
    )
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Build figure
    i = 0
    for row in tqdm(ax, desc='Building png'):
        for col in row:

            col.set_xticklabels([])
            col.set_yticklabels([])
            col.set_aspect('equal')
            col.axis("off")

            # Display the background image
            if z0+i < img.shape[2]:
                col.imshow(
                    img[x0:x1+1, y0:y1+1, z0+i].T, 
                    cmap='gray', 
                    interpolation='none', 
                    vmin=0, 
                    vmax=np.mean(img) + 2 * np.std(img),
                )

            # Overlay each mask
            if z0+i <= z1:
                for mask, color in zip(masks, colors):
                    rgba = np.zeros((x1+1-x0, y1+1-y0, 4), dtype=float)
                    for c in range(4):  # RGBA
                        rgba[..., c] = mask[x0:x1+1, y0:y1+1, z0+i] * color[c]
                    col.imshow(rgba.transpose((1,0,2)), interpolation='none')

            i += 1

    # fig.suptitle('Mask overlay', fontsize=14)
    fig.savefig(file, bbox_inches='tight', pad_inches=0)
    #plt.savefig(file)
    plt.close()