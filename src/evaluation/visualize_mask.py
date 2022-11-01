# %%
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import argparse
import torch
import numpy as np
from tqdm import tqdm
import open3d as o3d

from nnutils.chamfer_distance import ChamferDistance 


def visualize_occlusion_mask(occlusion_mask, world2grid):
    dim_x = occlusion_mask.shape[0]
    dim_y = occlusion_mask.shape[1]
    dim_z = occlusion_mask.shape[2]

    # Generate voxel indices.
    x = torch.arange(dim_x, dtype=occlusion_mask.dtype, device=occlusion_mask.device)
    y = torch.arange(dim_y, dtype=occlusion_mask.dtype, device=occlusion_mask.device)
    z = torch.arange(dim_z, dtype=occlusion_mask.dtype, device=occlusion_mask.device)

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
    grid_xyz = torch.cat([
        grid_x.view(dim_x, dim_y, dim_z, 1),
        grid_y.view(dim_x, dim_y, dim_z, 1),
        grid_z.view(dim_x, dim_y, dim_z, 1)
    ], dim=3)

    # Filter visible points.
    grid_xyz = grid_xyz[occlusion_mask > 0.5]
    num_occluded_voxels = grid_xyz.shape[0]

    # Transform voxels to world space.
    grid2world = torch.inverse(world2grid)
    R_grid2world = grid2world[:3, :3].view(1, 3, 3).expand(num_occluded_voxels, -1, -1)
    t_grid2world = grid2world[:3, 3].view(1, 3, 1).expand(num_occluded_voxels, -1, -1)
    
    grid_xyz_world = (torch.matmul(R_grid2world, grid_xyz.view(-1, 3, 1)) + t_grid2world).view(-1, 3)
    
    return grid_xyz_world


def filter_occluded_points(points_pred, world2grid, occlusion_mask):
    dim_x = occlusion_mask.shape[0]
    dim_y = occlusion_mask.shape[1]
    dim_z = occlusion_mask.shape[2]
    num_points_pred = points_pred.shape[0]

    # Transform points to bbox space.
    R_world2grid = world2grid[:3, :3].view(1, 3, 3).expand(num_points_pred, -1, -1)
    t_world2grid = world2grid[:3, 3].view(1, 3, 1).expand(num_points_pred, -1, -1)
    
    points_pred_coords = (torch.matmul(R_world2grid, points_pred.view(num_points_pred, 3, 1)) + t_world2grid).view(num_points_pred, 3)

    # Normalize to [-1, 1]^3 space.
    # The world2grid transforms world positions to voxel centers, so we need to
    # use "align_corners=True".
    points_pred_coords[:, 0] /= (dim_x - 1)
    points_pred_coords[:, 1] /= (dim_y - 1)
    points_pred_coords[:, 2] /= (dim_z - 1)
    points_pred_coords = points_pred_coords * 2 - 1

    # Trilinearly interpolate occlusion mask.
    # Occlusion mask is given as (x, y, z) storage, but the grid_sample method
    # expects (c, z, y, x) storage.
    visibility_mask = 1 - occlusion_mask.view(dim_x, dim_y, dim_z)
    visibility_mask = visibility_mask.permute(2, 1, 0).contiguous()
    visibility_mask = visibility_mask.view(1, 1, dim_z, dim_y, dim_x)

    points_pred_coords = points_pred_coords.view(1, 1, 1, num_points_pred, 3)

    points_pred_visibility = torch.nn.functional.grid_sample(
        visibility_mask, points_pred_coords.cpu(), mode='bilinear', padding_mode='zeros', align_corners=True
    ).cuda()

    points_pred_visibility = points_pred_visibility.view(num_points_pred)

    eps = 1e-5
    points_pred_visibility = points_pred_visibility >= 1 - eps

    # Filter occluded predicted points.
    if points_pred_visibility.sum() == 0:
        # If no points are visible, we keep the original points, otherwise
        # we would penalize the sample as if nothing is predicted.
        print("All points occluded, keeping all predicted points!")
        points_pred_visible = points_pred.clone()
    else:
        points_pred_visible = points_pred[points_pred_visibility]

    return points_pred_visible

# %%
#####################################################################################
# Settings.
#####################################################################################
dist_threshold = 0.05
max_dist = 1.0
num_points_samples = 200000


groundtruth_dir = "/mnt/res_nas/mohameds/trans_fusion_data/groundtruth/"

assert os.path.exists(groundtruth_dir)


scene_ids = sorted(os.listdir(groundtruth_dir))

scene_id = "scene0708_00"


# Load groundtruth mesh.
mesh_gt_path = os.path.join(groundtruth_dir, scene_id, "mesh_gt.ply".format(scene_id))


occlusion_mask_path = os.path.join(groundtruth_dir, scene_id, "occlusion_mask.npy")
occlusion_mask = np.load(occlusion_mask_path)

world2grid_path = os.path.join(groundtruth_dir, scene_id, "world2grid.txt")
world2grid = np.loadtxt(world2grid_path)

# We keep occlusion mask on host memory, since it can be very large for big scenes.
occlusion_mask = torch.from_numpy(occlusion_mask).float().cuda()

# %%


mesh_gt = o3d.io.read_triangle_mesh(mesh_gt_path)
points_gt = np.asarray(mesh_gt.vertices)

# Just for debugging: Visualize occluded points.
occluded_pcd = o3d.geometry.PointCloud()
occluded_pcd.points = o3d.utility.Vector3dVector(visualize_occlusion_mask(occlusion_mask, world2grid).cpu().numpy())
occluded_pcd.paint_uniform_color([0.7, 0.0, 0.0])

o3d.visualization.draw_geometries([mesh_gt, occluded_pcd], mesh_show_back_face=True)
# %%
