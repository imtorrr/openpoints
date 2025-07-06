import itertools
import os
import glob
import pickle
import logging
import numpy as np
from tqdm import tqdm
import torch
import laspy
import threading
from torch.utils.data import Dataset
from torch_geometric.data import Data
from ..data_util import crop_pc, voxelize
from ..build import DATASETS


def slice_pts(pc, bx, by, bz, box_dim, min_points_per_box=1200, max_points_per_box=24000, ratio=5):
    pc_slice = pc[
        (pc[:, 0] >= bx)
        & (pc[:, 0] <= bx + box_dim)
        & (pc[:, 1] >= by)
        & (pc[:, 1] <= by + box_dim)
        & (pc[:, 2] >= bz)
        & (pc[:, 2] <= bz + box_dim)
    ]

    if len(pc_slice) > min_points_per_box:
        if len(pc_slice) > max_points_per_box * ratio:
            indices = np.random.choice(
                pc_slice.shape[0], size=max_points_per_box * ratio, replace=False
            )
            pc_slice = pc_slice[indices]

        return pc_slice


def tile_point_cloud(pc, box_dim: float = 6, box_overlap: float = 0.5):
    xmin, ymin, zmin = np.floor(np.min(pc[:, :3], axis=0))
    xmax, ymax, zmax = np.ceil(np.max(pc[:, :3], axis=0))

    box_overlap = box_dim * box_overlap

    x_cnr = np.arange(xmin - box_overlap, xmax + box_overlap, box_overlap)
    y_cnr = np.arange(ymin - box_overlap, ymax + box_overlap, box_overlap)
    z_cnr = np.arange(zmin - box_overlap, zmax + box_overlap, box_overlap)

    pc_sliced = []

    for bx, by, bz in itertools.product(x_cnr, y_cnr, z_cnr):
        result = slice_pts(pc, bx, by, bz, box_dim)
        if result is not None:
            pc_sliced.append(result)

    return pc_sliced


def save_tile(arr, out_path):
    np.save(out_path, arr)


@DATASETS.register_module()
class NIBIO_MLS(Dataset):
    classes = ["ground", "vegetation", "cwd", "stem"]
    num_classes = len(classes)
    num_per_class = np.array(
        [
            38944970,
            88990151,
            463191,
            19693742,
        ],
        dtype=np.int32,
    )
    class2color = {"ground": [0, 0, 255], "vegetation": [0, 255, 0], "cwd": [0, 255, 255], "stem": [255, 0, 0]}
    cmap = [*class2color.values()]
    gravity_dim = 2

    def __init__(
        self,
        data_root: str = "data/nibio_mls/",
        tile_size: float = 6,
        voxel_size: float = 0.04,
        as_pyg:bool = True,
        split: str = "train",
        loop: int = 1,
        transform=None,
        presample: bool = False,
        shuffle: bool = True,
    ):
        super().__init__()
        self.split, self.voxel_size, self.transform = split, voxel_size, transform
        self.presample = presample
        self.shuffle = shuffle
        self.loop = loop
        self.as_pyg = as_pyg

        self.raw_root = os.path.join(data_root, "raw")
        self.raw_list = glob.glob(os.path.join(self.raw_root, split, "*"))

        self.tiled_root = os.path.join(data_root, "tiled", split)
        os.makedirs(self.tiled_root, exist_ok=True)
        processed_root = os.path.join(data_root, "processed")
        filename = os.path.join(
            processed_root, f"nibio_mls_{split}_{voxel_size:.3f}_{tile_size}m.pkl"
        )
        if presample and not os.path.exists(filename):
            np.random.seed(0)
            self.data = []
            for path in tqdm(self.raw_list, desc=f"Loading NIBIO_MLS {split} split"):
                las = laspy.read(path)
                pc = np.vstack(
                    [np.array(las.x), np.array(las.y), np.array(las.z), np.array(las.label)]
                ).T
                pc = pc.astype(np.float32)
                pc[:, :3] -= np.min(pc[:, :3], 0)
                if voxel_size:
                    uniq_idx = voxelize(pc[:, :3], voxel_size)
                    pc = pc[uniq_idx]

                pc_sliced = tile_point_cloud(pc, box_dim=tile_size)
                self.data += pc_sliced

                threads = []

                name = os.path.splitext(os.path.basename(path))[0]
                for i, arr in enumerate(pc_sliced):
                    out_path = os.path.join(self.tiled_root, f"{name}-{i:08d}")

                    threads.append(threading.Thread(target=save_tile, args=(arr, out_path)))

                for x in threads:
                    x.start()
                for x in threads:
                    x.join()

            npoints = np.array([len(data) for data in self.data])
            logging.info(
                "split: %s, median npoints %.1f, avg num points %.1f, std %.1f"
                % (self.split, np.median(npoints), np.average(npoints), np.std(npoints))
            )
            os.makedirs(processed_root, exist_ok=True)
            with open(filename, "wb") as f:
                pickle.dump(self.data, f)
                print(f"{filename} saved successfully")
        elif presample:
            with open(filename, "rb") as f:
                self.data = pickle.load(f)
                print(f"{filename} load successfully")
        self.data_list = glob.glob(os.path.join(self.tiled_root, "*.npy"))
        self.data_idx = np.arange(len(self.data_list))
        assert len(self.data_idx) > 0
        logging.info(f"\nTotally {len(self.data_idx)} sample in {split} set")

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        if self.presample:
            coord, label = np.split(self.data[data_idx], [3], axis=1)
        else:
            data_path = self.data_list[data_idx]
            cdata = np.load(data_path).astype(np.float32)
            coord, label = np.split(cdata, [3], axis=1)

        coord -= np.min(coord, 0)
        label = label.squeeze(-1).astype(np.long)
        data = {"pos": coord, "y": label}

        if self.transform is not None:
            data = self.transform(data)

        if "heights" not in data.keys():
            data["heights"] = torch.from_numpy(
                coord[:, self.gravity_dim : self.gravity_dim + 1].astype(np.float32)
            )
        
        data["x"] = data["pos"]
        
        if self.as_pyg:
            data = Data.from_dict(data)
        
        return data

    def __len__(self):
        return len(self.data_idx) * self.loop
