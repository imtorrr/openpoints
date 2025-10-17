import numpy as np
import os
import laspy
import logging
from glob import glob
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle

from ..build import DATASETS
from ..data_util import crop_pc, tile_pc_fast

@DATASETS.register_module()
class NIBIO_MLS(Dataset):
    classes = [
        "ground",
        "vegetation",
        "cwd",
        "stem",
    ]

    num_classes = 4
    num_per_class = np.array(
        [
            38944970,
            88990151,
            463191,
            19693742,
        ],
        dtype=np.int32,
    )
    class2color = {
        "ground": [0, 0, 255],
        "vegetation": [0, 255, 0],
        "cwd": [0, 255, 255],
        "stem": [255, 0, 0],
    }
    cmap = [*class2color.values()]
    gravity_dim = 2

    def __init__(
        self,
        data_root: str = "data/NIBIO_MLS/",
        voxel_size: float = 0.04,
        voxel_max: int | None = None,
        split: str = "train",
        transform=None,
        loop: int = 1,
        presample: bool = False,
        variable: bool = False,
        shuffle: bool = True,
    ):
        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.loop = (
            split,
            voxel_size,
            transform,
            voxel_max,
            loop,
        )
        self.presample = presample
        self.variable = variable
        self.shuffle = shuffle

        raw_root = os.path.join(data_root, "raw")
        self.raw_root = raw_root
        self.data_list = glob(os.path.join(raw_root, split, "*"))

        tiled_root = os.path.join(data_root, "tiled", split)

        # ensure tiled directory exists
        if not os.path.isdir(tiled_root):
            os.makedirs(tiled_root, exist_ok=True)

        # ensure tiled directory contain data
        if len(os.listdir(tiled_root)) == 0:
            for data_path in tqdm(self.data_list, desc=f"Tiling NIBIO_MLS {split} split"):
                las = laspy.read(data_path)
                pc = np.vstack([las.x, las.y, las.z, las.label]).T
                tiles = tile_pc_fast(pc, box_overlap=0)

                basename = os.path.basename(data_path)
                filename, fmt = os.path.splitext(basename)

                for i, tile in enumerate(tiles):
                    tile_name = filename + f"_{i}"
                    tile_path = os.path.join(tiled_root, tile_name)

                    np.save(tile_path, tile)

        self.data_list = glob(os.path.join(tiled_root, "*.npy"))

        processed_root = os.path.join(data_root, "processed")
        filename = os.path.join(
            processed_root, f"nibio-mls_{split}_{voxel_size:.3f}_{voxel_max}.pkl"
        )

        if presample and not os.path.exists(filename):
            np.random.seed(0)
            self.data = []
            for data_path in tqdm(self.data_list, desc=f"Loading NIBIO_MLS {split} split"):
                pc = np.load(data_path).astype(np.float32)
                pc[:, :3] -= np.min(pc[:, :3], axis=0)
                if voxel_size:
                    coord, feat, label = pc[:, :3], None, pc[:, 3:4]
                    coord, feat, label = crop_pc(
                        coord,
                        feat,
                        label,
                        self.split,
                        self.voxel_size,
                        self.voxel_max,
                        downsample=True,
                        variable=self.variable,
                        shuffle=self.shuffle,
                    )

                    pc = np.hstack((coord, label)).astype(np.float32)
                self.data.append(pc)
            npoints = np.array([len(data) for data in self.data])
            logging.info(
                "split: %s, median npoints %.1f, avg num points %.1f, std %.1f"
                % (self.split, np.median(npoints), np.average(npoints), np.std(npoints))
            )
            os.makedirs(processed_root, exist_ok=True)
            with open(filename, "wb") as f:
                pickle.dump(self.data, f)
                logging.info(f"{filename} saved successfully")
        elif presample:
            with open(filename, "rb") as f:
                self.data = pickle.load(f)
                logging.info(f"{filename} load successfully")

        self.data_idx = np.arange(len(self.data_list))
        assert len(self.data_idx) > 0
        logging.info(f"\nTotally {len(self.data_idx)} samples in {split} set")

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        feat = None
        if self.presample:
            coord, label = np.split(self.data[data_idx], [3], axis=1)
        else:
            data_path = self.data_list[data_idx]
            pc = np.load(data_path).astype(np.float32)
            pc[:, :3] -= np.min(pc[:, :3], axis=0)
            coord, label = np.split(pc, [3], axis=1)
            coord, _, label = crop_pc(
                coord,
                None,
                label,
                self.split,
                self.voxel_size,
                self.voxel_max,
                downsample=not self.presample,
                variable=self.variable,
                shuffle=self.shuffle,
            )
            
        label = label.squeeze(-1).astype(np.int64)
        data = {"pos": coord, "x": feat, "y": label}
        
        if self.transform is not None:
            data = self.transform(data)
            
        if "heights" not in data.keys():
            data["heights"] = torch.from_numpy(
                coord[:, self.gravity_dim : self.gravity_dim + 1].astype(np.float32)
            )
        return data

    def __len__(self):
        return len(self.data_idx) * self.loop