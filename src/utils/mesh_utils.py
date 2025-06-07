"""
Mesh processing utilities for 3D surface meshes.
"""

import trimesh
import numpy as np


def _fix_degenerate(mesh: trimesh.Trimesh):
    """Fix degenerate triangles in a mesh.
    
    Args:
        mesh: Input mesh
        
    Returns:
        Fixed mesh
    """
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
    mesh.update_faces(
        np.setdiff1d(np.arange(len(mesh.faces)), zero_area_indices)
    )
    mesh.remove_unreferenced_vertices()

    # Fix any potential inversion in the mesh
    mesh.fix_normals()
    return mesh


def fix_degenerate_mesh_np(vert, face, normal):
    """Fix degenerate triangles in a mesh given as numpy arrays.
    
    Args:
        vert: Vertex coordinates
        face: Face indices
        normal: Vertex normals
        
    Returns:
        Tuple of (fixed vertices, faces, normals)
    """
    # Create a trimesh object from the input vertices and faces
    mesh = trimesh.Trimesh(vertices=vert, faces=face, vertex_normals=normal)

    # Fix degenerate mesh using the _fix_degenerate function
    fixed_mesh = _fix_degenerate(mesh)

    # Extract the fixed vertices, faces, and normals from the fixed mesh
    fixed_vert = fixed_mesh.vertices
    fixed_face = fixed_mesh.faces
    fixed_normal = fixed_mesh.vertex_normals

    return fixed_vert, fixed_face, fixed_normal


def fix_degenerate_files(in_mesh_fn, out_mesh_fn):
    """Fix degenerate triangles in a mesh file.
    
    Args:
        in_mesh_fn: Input mesh file path
        out_mesh_fn: Output mesh file path
    """
    # Load the mesh using trimesh
    mesh = trimesh.load(in_mesh_fn)
    mesh = _fix_degenerate(mesh)
    mesh.export(out_mesh_fn) 