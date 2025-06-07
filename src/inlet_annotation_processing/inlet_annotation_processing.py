import argparse
import glob
import os

import numpy as np
import tqdm
import trimesh
from skimage.measure import marching_cubes

import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)
from src.utils.file_io import load_nii, read_json, save_nii


def vol_idx_to_pcd_transform(vol_idx):
    vert = vol_idx / 256 - 0.5
    return vert


def pcd_to_vol_idx_transform(pcd):
    vert = (pcd + 0.5) * 256
    vert = vert.astype(np.int32)
    return vert


def gen_ply_for_annotataion(imagecas_fns, ply_fns):
    for imagecas_label_fn, ply_fn in tqdm.tqdm(zip(imagecas_fns, ply_fns)):
        v, info = load_nii(imagecas_label_fn)
        verts, faces, normals, _ = marching_cubes(v, 0.5)
        sampledverts = verts[np.random.choice(np.arange(len(verts)), 5000), :]
        pcd_vert = vol_idx_to_pcd_transform(sampledverts)
        point_cloud = trimesh.PointCloud(vertices=pcd_vert)
        point_cloud.export(ply_fn)


def L_R_position_from_annotation(ann):

    L = None
    R = None
    for keypoint in ann["keypoints"]:
        if keypoint["class_id"] == 0:
            L = pcd_to_vol_idx_transform(
                np.array(keypoint["position"], dtype=np.float32)
            )
        elif keypoint["class_id"] == 1:
            R = pcd_to_vol_idx_transform(
                np.array(keypoint["position"], dtype=np.float32)
            )
        else:
            print("error annotation")
    return L.round().astype(np.int32), R.round().astype(np.int32)


def parse_annotation_gen_volume(imagecas_fns, json_fns, inlet_fns):

    for imagecas_fn, annotation_fn, inlet_fn in tqdm.tqdm(
        zip(imagecas_fns, json_fns, inlet_fns)
    ):
        v, info = load_nii(imagecas_fn)
        ann = read_json(annotation_fn)
        L, R = L_R_position_from_annotation(ann)
        root_volume = np.zeros_like(v)
        root_volume[L[0], L[1], L[2]] = 1
        root_volume[R[0], R[1], R[2]] = 2
        save_nii(root_volume, inlet_fn, ref_nii_path=imagecas_fn)


if __name__ == "__main__":
    """
    example:
    python inlet_annotataion_processing.py --pre --imagecas_root ./data/imagecas --annotation_root ./data/inlet_annotation
    python inlet_annotataion_processing.py --post --imagecas_root ./data/imagecas --annotation_root ./data/inlet_annotation
    """
    parser = argparse.ArgumentParser(
        description="pre/post process of annotation"
    )
    parser.add_argument(
        "--pre", action="store_true", help="Preprocess the annotation"
    )
    parser.add_argument(
        "--post", action="store_true", help="Preprocess the annotation"
    )
    parser.add_argument(
        "--imagecas_root", type=str, help="Root path of the data"
    )
    parser.add_argument("--annotation_root", type=str, help="Save location")

    args = parser.parse_args()

    imagecas_fns = sorted(
        glob.glob(os.path.join(args.imagecas_root, "*.label.nii.gz"))
    )
    ply_annotation_fns = [
        f"{args.annotation_root}/{os.path.basename(fn)}.ply"
        for fn in imagecas_fns
    ]
    json_annotation_fns = [
        f"{args.annotation_root}/{os.path.basename(fn)}.json"
        for fn in imagecas_fns
    ]
    inlet_volume_fns = [
        f"{args.annotation_root}/{os.path.basename(fn)}" for fn in imagecas_fns
    ]

    if args.pre:
        os.makedirs(args.annotation_root, exist_ok=True)
        gen_ply_for_annotataion(imagecas_fns, ply_annotation_fns)
    elif args.post:
        parse_annotation_gen_volume(
            imagecas_fns, json_annotation_fns, inlet_volume_fns
        )
