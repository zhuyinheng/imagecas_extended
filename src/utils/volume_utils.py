"""
Volume processing utilities for medical image analysis.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from skimage.measure import marching_cubes, regionprops, label


def vertices_world_to_volume(v, origin_world, voxel_spacing):
    """Convert world coordinates to volume coordinates.
    
    Args:
        v: World coordinates
        origin_world: World origin
        voxel_spacing: Voxel spacing
        
    Returns:
        Volume coordinates
    """
    voxel_idx = (v - origin_world) / voxel_spacing
    return voxel_idx


def mask_mass_center(volume):
    """Calculate the mass center of a binary mask.
    
    Args:
        volume: Binary mask volume
        
    Returns:
        Mass center coordinates
    """
    H, W, D = volume.shape
    # grid_pts: HxWxDx3
    grid_pts = np.stack(
        np.meshgrid(
            np.linspace(0, H - 1, H, dtype=np.float32),
            np.linspace(0, W - 1, W, dtype=np.float32),
            np.linspace(0, D - 1, D, dtype=np.float32),
            indexing="ij",
        ),
        -1,
    )
    return (volume[..., None] * grid_pts).reshape(H * W * D, 3).sum(0)


def connected_conponent_sorted_by_area(volume):
    """Label connected components in a binary volume and sort by area.
    
    Args:
        volume: Binary volume
        
    Returns:
        Labeled volume with components sorted by area
    """
    _label_volume = label(volume)
    props = regionprops(_label_volume)
    num_conn_conp = len(props)
    areas = -np.ones(num_conn_conp)
    for ia in range(num_conn_conp):
        areas[ia] = props[ia].area
    index = areas.argsort()[::-1]
    label_v = np.zeros_like(_label_volume)
    for i, idx in enumerate(index):
        label_v[_label_volume == idx + 1] = i + 1
    return label_v


def pad_volume(v, pad=7):
    """Pad a volume with zeros.
    
    Args:
        v: Input volume
        pad: Padding size
        
    Returns:
        Padded volume
    """
    new_v = np.zeros(np.array(v.shape) + 2 * pad, dtype=v.dtype)
    new_v[pad:-pad, pad:-pad, pad:-pad] = v
    return new_v


def mask_volume_bbox(v):
    """Get bounding box of non-zero elements in a volume.
    
    Args:
        v: Input volume
        
    Returns:
        Bounding box coordinates [min, max]
    """
    pts = np.stack(np.where(v > 0), -1)
    return np.stack([pts.min(0), pts.max(0)], 0)


def mask_volume_list_bbox(v_list):
    """Get bounding box of non-zero elements in a list of volumes.
    
    Args:
        v_list: List of volumes
        
    Returns:
        Bounding box coordinates [min, max]
    """
    # v in v_list should has same shape
    for v in v_list:
        assert v.shape == v_list[0].shape
    bbox_list = []
    for v in v_list:
        bbox_list.append(mask_volume_bbox(v))
    # bbox[0]=[x_min,y_min,z_min], bbox[1]=[x_max,y_max,z_max]
    bbox_list = np.stack(bbox_list, 0)
    return np.stack([bbox_list[:, 0, :].min(0), bbox_list[:, 1, :].max(0)], 0)


def non_zero_sub_volume(v):
    """Extract sub-volume containing non-zero elements.
    
    Args:
        v: Input volume
        
    Returns:
        Tuple of (sub-volume, bbox_min)
    """
    bbox = mask_volume_bbox(v)
    return (
        v[
            bbox[0, 0] : bbox[1, 0] + 1,
            bbox[0, 1] : bbox[1, 1] + 1,
            bbox[0, 2] : bbox[1, 2] + 1,
        ],
        bbox[0],
    )


def non_zero_sub_list_volume(v_list):
    """Extract sub-volumes containing non-zero elements from a list.
    
    Args:
        v_list: List of volumes
        
    Returns:
        Tuple of (list of sub-volumes, bbox_min)
    """
    bbox = mask_volume_list_bbox(v_list)
    return [
        v[
            bbox[0, 0] : bbox[1, 0] + 1,
            bbox[0, 1] : bbox[1, 1] + 1,
            bbox[0, 2] : bbox[1, 2] + 1,
        ]
        for v in v_list
    ], bbox[0]


def non_zero_sub_mask(volume, voxel_spacing, origin_world, padding=7):
    """Extract sub-volume containing non-zero elements with padding.
    
    Args:
        volume: Input volume
        voxel_spacing: Voxel spacing
        origin_world: World origin
        padding: Padding size
        
    Returns:
        Tuple of (padded sub-volume, new origin)
    """
    _volume, bbox_min = non_zero_sub_volume(volume)
    _volume = pad_volume(_volume, pad=padding)
    bbox_min -= padding
    _origin_world = origin_world + bbox_min * voxel_spacing
    return (_volume, _origin_world)


def resize_volume_in_shape(volume, new_shape, method="nn"):
    """Resize a 3D volume to a new shape.
    
    Args:
        volume: Input volume
        new_shape: Desired shape
        method: Interpolation method ("nn" or "linear")
        
    Returns:
        Resized volume
    """
    # Calculate the zoom factors for each dimension
    zoom_factors = [n / o for n, o in zip(new_shape, volume.shape)]

    if method == "nn":
        order = 0  # Nearest neighbor interpolation
    elif method == "linear":
        order = 1  # Linear interpolation
    else:
        raise ValueError(
            "Invalid method. Choose 'nn' for nearest neighbor or 'linear' for"
            " linear interpolation."
        )

    # Resize the volume
    resized_volume = zoom(volume, zoom_factors, order=order)

    return resized_volume


def surface_from_mask_with_crop(
    volume,
    voxel_spacing,
    origin_world,
    interp_voxel_spacing=0.3,
    interp_padding=7,
):
    """Generate surface mesh from a binary mask with cropping.
    
    Args:
        volume: Binary mask
        voxel_spacing: Voxel spacing
        origin_world: World origin
        interp_voxel_spacing: Target voxel spacing
        interp_padding: Padding size
        
    Returns:
        Tuple of (vertices, faces, normals)
    """
    _volume, _origin_world = non_zero_sub_mask(
        volume, voxel_spacing, origin_world, padding=interp_padding
    )
    _volume = resize_volume_in_shape(
        _volume,
        np.round(np.array(_volume.shape) * voxel_spacing / interp_voxel_spacing).astype(int),
        method="nn",
    )
    v, f, n, _ = marching_cubes(_volume > 0, 0, allow_degenerate=False)
    v = v * interp_voxel_spacing + _origin_world
    return v, f, n 