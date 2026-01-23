import numpy as np
import torch
from torch_geometric.data import Data
import os
import os.path as osp
import ssl
import sys
import urllib
import h5py
import itertools
from typing import Optional


class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in [".npy"]:
            return cls._read_npy(file_path)
        elif file_extension in [".pcd"]:
            return cls._read_pcd(file_path)
        elif file_extension in [".h5"]:
            return cls._read_h5(file_path)
        elif file_extension in [".txt"]:
            return cls._read_txt(file_path)
        else:
            raise Exception("Unsupported file extension: %s" % file_extension)

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)

    # # References: https://github.com/dimatura/pypcd/blob/master/pypcd/pypcd.py#L275
    # # Support PCD files without compression ONLY!
    # @classmethod
    # def _read_pcd(cls, file_path):
    #     pc = open3d.io.read_point_cloud(file_path)
    #     ptcloud = np.array(pc.points)
    #     return ptcloud

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, "r")
        return f["data"][()]


# download
def download_url(
    url: str, folder: str, log: bool = True, filename: Optional[str] = None
):
    r"""Downloads the content of an URL to a specific folder.
    Borrowed from https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/download.py
    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    if filename is None:
        filename = url.rpartition("/")[2]
        filename = filename if filename[0] == "?" else filename.split("?")[0]

    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print(f"Using existing file {filename}", file=sys.stderr)
        return path

    if log:
        print(f"Downloading {url}", file=sys.stderr)

    os.makedirs(folder, exist_ok=True)
    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, "wb") as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path


def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Shift coordinates to positive before hashing to handle negative coordinates
    arr = arr.copy()
    arr -= arr.min(0)  # Subtract minimum to ensure all values are non-negative
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(
        arr.shape[0], dtype=np.uint64
    )
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def voxelize(
    coord,
    voxel_size=0.05,
    hash_type="fnv",
    mode=0,
    offset: np.ndarray = np.array([0.0, 0.0, 0.0]),
):
    discrete_coord = np.floor((coord + offset) / np.array(voxel_size))
    if hash_type == "ravel":
        key = ravel_hash_vec(discrete_coord)
    else:
        key = fnv_hash_vec(discrete_coord)

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, voxel_idx, count = np.unique(
        key_sort, return_counts=True, return_inverse=True
    )
    if mode == 0:  # train mode
        idx_select = (
            np.cumsum(np.insert(count, 0, 0)[0:-1])
            + np.random.randint(0, count.max(), count.size) % count
        )
        idx_unique = idx_sort[idx_select]
        return idx_unique
    else:  # val mode
        return idx_sort, voxel_idx, count


def crop_pc(
    coord,
    feat,
    label,
    split="train",
    voxel_size=0.04,
    voxel_max=None,
    downsample=True,
    variable=True,
    shuffle=True,
):
    if voxel_size and downsample:
        # Is this shifting a must? I borrow it from Stratified Transformer and Point Transformer.
        coord -= coord.min(0)
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = (
            coord[uniq_idx],
            feat[uniq_idx] if feat is not None else None,
            label[uniq_idx] if label is not None else None,
        )
    if voxel_max is not None:
        crop_idx = None
        N = len(label)  # the number of points
        if N >= voxel_max:
            init_idx = np.random.randint(N) if "train" in split else N // 2
            crop_idx = np.argsort(
                np.sum(np.square(coord - coord[init_idx]), 1)
            )[:voxel_max]
        elif not variable:
            # fill more points for non-variable case (batched data)
            cur_num_points = N
            query_inds = np.arange(cur_num_points)
            padding_choice = np.random.choice(
                cur_num_points, voxel_max - cur_num_points
            )
            crop_idx = np.hstack([query_inds, query_inds[padding_choice]])
        crop_idx = np.arange(coord.shape[0]) if crop_idx is None else crop_idx
        if shuffle:
            shuffle_choice = np.random.permutation(np.arange(len(crop_idx)))
            crop_idx = crop_idx[shuffle_choice]
        coord, feat, label = (
            coord[crop_idx],
            feat[crop_idx] if feat is not None else None,
            label[crop_idx] if label is not None else None,
        )
    coord -= coord.min(0)
    return (
        coord.astype(np.float32),
        feat.astype(np.float32) if feat is not None else None,
        label.astype(np.long) if label is not None else None,
    )


def tile_pc(
    pc,
    box_dim: float = 6.0,
    box_overlap: float = 0.5,
):
    xmin, ymin = np.floor(np.min(pc[:, :2], axis=0))
    xmax, ymax = np.ceil(np.max(pc[:, :2], axis=0))
    zmin, zmax = np.min(pc[:, 2]), np.max(pc[:, 2])
    overlap_dist = box_dim * box_overlap
    stride = box_dim - overlap_dist
    x_edges = np.arange(xmin - stride, xmax + stride, stride)
    y_edges = np.arange(ymin - stride, ymax + stride, stride)
    z_edges = np.arange(zmin - stride, zmax + stride, stride)
    tiles = []
    boxes = []
    for x0, y0, z0 in itertools.product(x_edges, y_edges, z_edges):
        x1 = x0 + box_dim
        y1 = y0 + box_dim
        z1 = z0 + box_dim
        mask = (
            (pc[:, 0] >= x0)
            & (pc[:, 0] < x1)
            & (pc[:, 1] >= y0)
            & (pc[:, 1] < y1)
            & (pc[:, 2] >= z0)
            & (pc[:, 2] < z1)
        )
        if not np.any(mask):
            continue
        tiles.append(pc[mask])
        boxes.append((x0, x1, y0, y1, z0, z1))

    return tiles


def tile_pc_fast(
    pc,
    las_path: str,
    output_dir: str,
    box_dim: float = 6.0,
    box_overlap: float = 0.5,
    voxel_max: Optional[int] = None,
    min_points_per_tile: int = 2000,
):
    overlap_dist = box_dim * box_overlap
    stride = box_dim - overlap_dist
    shifts = []
    # generate shift per axis "xyz"
    for _ in range(3):
        shifts.append(np.arange(0.0, box_dim, stride))
    offset_list = [np.array(o, dtype=float) for o in itertools.product(*shifts)]
    for off in offset_list:
        idx_sort, voxel_idx, count = voxelize(
            pc[:, :3], voxel_size=box_dim, mode=1, offset=off
        )
        pc_sort = pc[idx_sort]  # points sorted by voxel
        # Fastest: split once using cumulative counts (no per-voxel boolean masks)
        cuts = np.cumsum(count)[:-1]  # split indices
        tile_list = np.split(pc_sort, cuts)  # list of arrays, one per voxel

        # Randomly sample points if tile exceeds voxel_max
        if voxel_max is not None:
            tile_list = [
                tile[np.random.choice(len(tile), voxel_max, replace=False)]
                if len(tile) > voxel_max
                else tile
                for tile in tile_list
            ]

        basename = os.path.basename(las_path)
        filename, _ = os.path.splitext(basename)

        for i, tile in enumerate(tile_list):
            if (
                len(tile) < min_points_per_tile
            ):  # Skip tiles with too few points
                continue
            tile_name = f"{filename}_{off[0]}_{off[1]}_{off[2]}_{i}"
            tile_path = os.path.join(output_dir, tile_name)
            np.save(tile_path, tile)


def get_features_by_keys(data, keys="pos,x"):
    key_list = keys.split(",")
    tensors = []
    for key in key_list:
        try:
            tensors.append(data[key.strip()])
        except (KeyError, AttributeError):
            raise ValueError(f"Key '{key}' not found in data.")

    result = torch.cat(tensors, dim=-1) if len(tensors) > 1 else tensors[0]

    if isinstance(data, dict):
        return result.transpose(1, 2).contiguous()
    elif isinstance(data, Data):
        return result.contiguous()


def get_class_weights(num_per_class, normalize=False):
    weight = num_per_class / float(sum(num_per_class))
    ce_label_weight = 1 / (weight + 0.02)

    if normalize:
        ce_label_weight = (
            ce_label_weight * len(ce_label_weight)
        ) / ce_label_weight.sum()
    return torch.from_numpy(ce_label_weight.astype(np.float32))


def get_class_alpha(num_per_class):
    inverse_freq = 1.0 / num_per_class

    alpha = inverse_freq / inverse_freq.sum()
    return torch.from_numpy(alpha.astype(np.float32))
