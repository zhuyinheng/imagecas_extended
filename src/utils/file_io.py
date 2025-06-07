"""
File I/O utilities for medical image processing.
"""

import json
import numpy as np
import SimpleITK as sitk
import trimesh


def write_triangle_surface_mesh(fn, v, f, n, check_surface=False):
    """Write a triangle surface mesh to file.
    
    Args:
        fn: Output file path
        v: Vertices array
        f: Faces array
        n: Vertex normals array
        check_surface: Whether to check and fix surface issues
    """
    mesh = trimesh.Trimesh(vertices=v, faces=f, vertex_normals=n)

    if check_surface:
        # Define a function to calculate triangle areas
        def triangle_areas(mesh):
            triangles = mesh.faces
            vertices = mesh.vertices
            vec_cross = np.cross(
                vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
            )
            areas = 0.5 * np.linalg.norm(vec_cross, axis=1)
            return areas

        # Calculate areas of the triangles
        areas = triangle_areas(mesh)

        # Find indices of zero-area triangles
        zero_area_indices = np.where(areas == 0)[0]

        # Remove zero-area triangles
        mesh.update_faces(np.setdiff1d(np.arange(len(mesh.faces)), zero_area_indices))
        mesh.remove_unreferenced_vertices()

        # Fix any potential inversion in the mesh
        mesh.fix_normals()

    # Export the cleaned mesh
    mesh.export(fn)


def read_json(fn):
    """Read JSON file.
    
    Args:
        fn: Input file path
        
    Returns:
        Loaded JSON data
    """
    with open(fn, "r") as f:
        return json.load(f)


def write_json(fn, x):
    """Write data to JSON file.
    
    Args:
        fn: Output file path
        x: Data to write
    """
    with open(fn, "w") as f:
        return json.dump(x, f, indent=4)


def load_nii(path, info_only=False, raw_only=False, inversion=False):
    """Load NIfTI file.
    
    Args:
        path: Input file path
        info_only: Whether to return only metadata
        raw_only: Whether to return only raw data
        inversion: Whether to invert the image
        
    Returns:
        Tuple of (image data, metadata) or just metadata/raw data based on flags
    """
    sitk_image = sitk.ReadImage(path)
    infos = {
        "shape": np.asarray(sitk_image.GetSize(), dtype=np.int32)[::-1],
        "spacing": np.asarray(sitk_image.GetSpacing())[::-1],
        "origin": np.asarray(sitk_image.GetOrigin())[::-1],
    }
    image = sitk.GetArrayFromImage(sitk_image)
    if inversion:
        image = image[::-1, ::-1, ::-1]
    if raw_only:
        return sitk_image
    elif info_only:
        return infos
    else:
        return image, infos


def save_nii(array, save_path, infos=None, ref_nii_path=None):
    """Save data as NIfTI file.
    
    Args:
        array: Data to save
        save_path: Output file path
        infos: Metadata dictionary
        ref_nii_path: Reference NIfTI file path for metadata
    """
    assert (infos is None) ^ (ref_nii_path is None)

    # Create a SimpleITK Image from the numpy array
    sitk_image = sitk.GetImageFromArray(array)
    if not (infos is None):
        # Set the metadata
        sitk_image.SetSpacing(np.asarray(infos["spacing"])[::-1])
        sitk_image.SetOrigin(np.asarray(infos["origin"])[::-1])
        sitk_image.SetDirection(np.asarray(infos["direction"])[::-1])
    else:
        # Read the reference image to get metadata
        ref_sitk_image = sitk.ReadImage(ref_nii_path)

        # Preserve the metadata from the reference image
        sitk_image.CopyInformation(ref_sitk_image)

    # Save the image
    sitk.WriteImage(sitk_image, save_path) 