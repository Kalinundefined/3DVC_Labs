import numpy as np
import open3d as o3d

from sklearn.neighbors import KDTree as sklearn_KDTree


pcd = o3d.io.read_point_cloud('/home/jialin/repo22/3dvc/3dvc_1/assignment2/Problem3/bunny.ply')
vertices = np.asarray(pcd.points)
vertices_cnt = len(vertices)
vertex_normals = np.asarray(pcd.normals)

def get_eps(initial_eps=0.005, decay=0.95, threshold=0.999):
    eps = initial_eps
    while True:
        tree = sklearn_KDTree(vertices, leaf_size=10, metric='euclidean')
        pts_ind = np.arange(len(vertices))
        _, ind1 = tree.query(vertices + eps * vertex_normals, k=1)
        _, ind2 = tree.query(vertices - eps * vertex_normals, k=1)
        # compute the ratio that p_{i+\eps* N} = p_i
        ratio = sum([(pts_ind == ind1.squeeze()).sum() / len(pts_ind), (pts_ind == ind2.squeeze()).sum() / len(pts_ind)]) / 2
        if ratio > threshold:
            break
        print(f"ratio is {ratio * 100}% when eps = {eps}")
        eps *= decay

    return eps

def get_grid(eps = 0.01, voxel_size=0.01):
    voxel_grid= o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    grid_index = np.asarray([i.grid_index for i in voxel_grid.get_voxels()])
    grid_coord = np.asarray([voxel_grid.get_voxel_center_coordinate(i.grid_index) for i in voxel_grid.get_voxels()])
    all_vertices = np.concatenate([vertices, vertices + eps * vertex_normals, vertices - eps * vertex_normals])

    tree = sklearn_KDTree(all_vertices, leaf_size=10, metric='euclidean')
    dist, query_ind = tree.query(grid_coord, k=10)
    pts_for_each_coord = all_vertices[query_ind]
    dm = np.zeros_like(dist)
    dm[query_ind >= 2*len(vertices)] = -2 * eps
    dm[query_ind >= len(vertices)] += eps

    def get_basis(pts, rank=2):
        pts_list = [np.ones(pts.shape[:2])[...,np.newaxis]]
        if rank > 0:
            pts_list += [pts]
        if rank > 1:
            pts2  = [pts * np.roll(pts, i, axis=-1) for i in range(2)]
            pts_list += pts2
        basis = np.concatenate(pts_list, axis=-1)
        return basis
    
    def f(pts, dist, dm):
        h = 1.
        theta = np.exp(- dist*dist / h / h)
        A = np.sqrt(theta)[...,np.newaxis] * get_basis(pts)
        return np.stack([np.linalg.lstsq(A[i], dm[i], rcond=-1)[0] for i in range(len(A))])
    
    ret = f(pts_for_each_coord, dist, dm) * get_basis(grid_coord[:,np.newaxis, :]).squeeze()
    ret = ret.sum(axis=-1)
    grid_cnt = grid_index.max(axis=0).max() + 1
    np_grid = np.ones((grid_cnt, grid_cnt, grid_cnt)) * 100
    for idx, index in enumerate(grid_index):
        np_grid[index[0], index[1], index[2]] = ret[idx]

    return np_grid

eps = get_eps(threshold=0.99)
grid = get_grid(eps=eps)

import mcubes
vertices, triangles = mcubes.marching_cubes(grid, 0)

# Export the result to sphere.dae
mcubes.export_mesh(vertices, triangles, "bunny.dae", "Bunny")