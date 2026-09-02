import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.fftpack import dctn, idctn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans



# -------------------------
# SDF
# -------------------------
def sdf_from_mask(mask):
    """
    Compute signed distance field from binary mask.
    Positive outside, negative inside.
    """
    dist_out = distance_transform_edt(~mask)
    dist_in  = distance_transform_edt(mask)
    return dist_out - dist_in

# -------------------------
# DCT and Truncation
# -------------------------
def dct_sdf(sdf, norm="ortho"):
    """Forward 3D DCT of SDF."""
    return dctn(sdf, norm=norm)

def truncate_dct(coeffs, keep_shape):
    """
    Truncate DCT coefficients by keeping only a low-frequency cube.
    Args:
        coeffs: full 3D DCT array
        keep_shape: tuple (kx, ky, kz) of how many coeffs to keep
    """
    kx, ky, kz = keep_shape
    truncated = np.zeros_like(coeffs)
    truncated[:kx, :ky, :kz] = coeffs[:kx, :ky, :kz]
    return truncated

def reconstruct_from_dct(coeffs_trunc, norm="ortho"):
    """Inverse DCT reconstruction."""
    sdf_recon = idctn(coeffs_trunc, norm=norm)
    mask_recon = sdf_recon < 0
    return mask_recon, sdf_recon

def coeffs_from_mask(mask, keep_shape=None, normalize=False):
    mask = mask.astype(bool)
    coeffs = dctn(sdf_from_mask(mask))
    if keep_shape is not None:
        coeffs = truncate_dct(coeffs, keep_shape)
    if normalize:
        coeffs /= np.max(coeffs)
    return coeffs

def mask_from_coeffs(coeffs):
    return idctn(coeffs) < 0

# -------------------------
# High-level pipeline
# -------------------------
def compress(mask, keep_shape=(16,16,16)):
    """
    Full pipeline: mask → SDF → DCT → truncate → reconstruct.
    Returns:
        coeffs_trunc : truncated DCT coefficients
        sdf_recon    : reconstructed SDF
        mask_recon   : reconstructed binary mask
    """
    mask = np.array(mask, dtype=bool)
    coeffs = coeffs_from_mask(mask, keep_shape)
    mask_recon = mask_from_coeffs(coeffs)

    return mask_recon, coeffs

def smooth_mask(mask, keep_shape=(16,16,16)):
    """
    Full pipeline: mask → SDF → DCT → truncate → reconstruct.
    Returns:
        mask_recon   : reconstructed binary mask
    """
    mask = np.array(mask, dtype=bool)
    coeffs = coeffs_from_mask(mask, keep_shape)
    mask_recon = mask_from_coeffs(coeffs)
    return mask_recon


# -------------------------
# Dataset-level Pipeline
# -------------------------
def flatten_coeffs(coeffs_trunc):
    """
    Flatten truncated cube into 1D vector (consistent across shapes).
    """
    return coeffs_trunc.flatten()

def dataset_to_features(masks, keep_shape=(16,16,16), norm="ortho"):
    """
    Convert a list of masks into truncated DCT coefficient vectors.
    Args:
        masks: list of binary 3D numpy arrays
        keep_shape: cube size to retain (kx, ky, kz)
    Returns:
        features: shape (n_samples, n_features)
    """
    features = []
    for mask in masks:
        sdf = sdf_from_mask(mask)
        coeffs = dct_sdf(sdf, norm=norm)
        coeffs_trunc = truncate_dct(coeffs, keep_shape)
        feat = flatten_coeffs(coeffs_trunc)
        features.append(feat)
    return np.array(features)

def run_pca(features, n_components=10):
    """
    Run PCA on feature matrix.
    Returns reduced features and fitted PCA object.
    """
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(features)
    return reduced, pca

def classify_shapes(features_reduced, n_clusters=2, random_state=0):
    """
    Cluster shapes in PCA space.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features_reduced)
    return labels, kmeans



