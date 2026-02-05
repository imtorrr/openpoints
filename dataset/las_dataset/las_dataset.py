import numpy as np
import os
import laspy
import logging
from glob import glob
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle
import joblib
import sys

sys.path.append(".")
from openpoints.dataset.build import DATASETS
from openpoints.dataset.data_util import compute_hag, crop_pc, tile_pc_fast


# @DATASETS.register_module()
class LASDataset(Dataset):
    """
    Universal dataset for LAS/LAZ point clouds with automatic tiling.

    Directory structure:
        data_root/
          ├── train/*.las, *.laz
          ├── val/*.las, *.laz
          └── test/*.las, *.laz

    Features:
    - Auto-discovery of .las/.laz files in split folders
    - Automatic tiling with configurable size and overlap
    - Flexible label support (any LAS field or None for unlabeled data)
    - Optional features: RGB, intensity, return number, etc.
    - Automatic caching for fast reloading

    Args:
        data_root: Path to root folder containing train/val/test splits
        split: 'train', 'val', or 'test'
        tile_size: Size of tiles in meters (default: 6.0)
        tile_overlap: Overlap ratio between tiles 0-1 (default: 0.5)
        voxel_size: Voxel size for downsampling (default: 0.04)
        voxel_max: Maximum number of points per sample (default: None)
        min_points_per_tile: Minimum points required per tile to keep it (default: 2000)
        label_field: LAS field to use as labels - 'classification', 'label', 'user_data', etc.
                     Set to None for unlabeled data (default: 'classification')
        label_offset: Offset to apply to labels (e.g., -1 to map [1,2,3] to [0,1,2]) (default: 0)
        use_rgb: Include RGB colors as features (default: False)
        use_intensity: Include intensity as feature (default: False)
        use_return_number: Include return number info as features (default: False)
        transform: Optional transform to apply to samples
        loop: Number of loops through dataset per epoch (default: 1)
        presample: Whether to presample and cache all data (default: False)
        variable: Allow variable number of points (default: False)
        shuffle: Shuffle points within each sample (default: True)
    """

    gravity_dim = 2  # Z-axis

    def __init__(
        self,
        data_root: str = "data/LAS/",
        split: str = "train",
        tile_size: float = 6.0,
        tile_overlap: float = 0.5,
        voxel_size: float = 0.02,
        voxel_max: int | None = None,
        min_points_per_tile: int = 2000,
        label_field: str | None = "classification",
        label_offset: int = 0,
        use_approximate_hag: bool = False,
        use_rgb: bool = False,
        use_intensity: bool = False,
        use_return_number: bool = False,
        transform=None,
        loop: int = 1,
        presample: bool = False,
        variable: bool = False,
        shuffle: bool = True,
    ):
        super().__init__()
        self.split = split
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        self.min_points_per_tile = min_points_per_tile
        self.label_field = label_field
        self.label_offset = label_offset
        self.use_approximate_hag = use_approximate_hag
        self.use_rgb = use_rgb
        self.use_intensity = use_intensity
        self.use_return_number = use_return_number
        self.transform = transform
        self.loop = loop
        self.presample = presample
        self.variable = variable
        self.shuffle = shuffle

        # Setup paths
        processed_root = os.path.join(data_root, "processed")
        os.makedirs(processed_root, exist_ok=True)
        tiled_root = os.path.join(data_root, "tiled", split)
        os.makedirs(tiled_root, exist_ok=True)
        raw_root = os.path.join(data_root, "raw")
        self.raw_root = raw_root
        split_path = os.path.join(raw_root, split)

        # Path for processed cache
        var_suffix = "var" if variable else "fixed"
        processed_filename = os.path.join(
            processed_root,
            f"las_{split}_{voxel_size:.3f}_{voxel_max}_{var_suffix}.joblib",
        )

        # 1. Check for processed data first (if presample is enabled)
        if presample and os.path.exists(processed_filename):
            self.data = joblib.load(processed_filename)
            logging.info(f"Loaded processed data from {processed_filename}")
            # Use len(self.data) directly, don't need data_list
            self.data_list = []  # Not needed when loading from processed
            self.data_idx = np.arange(len(self.data))
        else:
            # 2. Check for tiled data
            tiled_files = glob(os.path.join(tiled_root, "*.npy"))

            if tiled_files:
                logging.info(f"Found {len(tiled_files)} tiled files")
                self.data_list = tiled_files
            else:
                # 3. Look for raw data and tile it
                if not os.path.exists(split_path):
                    raise ValueError(f"Split folder not found: {split_path}")

                # Find all .las and .laz files
                self.data_list = []
                self.data_list.extend(glob(os.path.join(split_path, "*.las")))
                self.data_list.extend(glob(os.path.join(split_path, "*.laz")))

                if len(self.data_list) == 0:
                    raise ValueError(f"No LAS/LAZ files found in {split_path}")

                logging.info(
                    f"Found {len(self.data_list)} LAS/LAZ files in {split} split"
                )

                # Tile the files
                logging.info(f"Tiling {len(self.data_list)} LAS/LAZ files...")
                for data_path in tqdm(
                    self.data_list, desc=f"Tiling LASDataset {split} split"
                ):
                    self._tile_las_file(data_path, tiled_root)

                # Load the tiled files
                self.data_list = glob(os.path.join(tiled_root, "*.npy"))

                if len(self.data_list) == 0:
                    raise ValueError(f"No tiled data found in {tiled_root}")

            # 4. Create processed cache if needed
            if presample:
                np.random.seed(0)
                self.data = []
                for data_path in tqdm(
                    self.data_list, desc=f"Loading LASDataset {split} split"
                ):
                    pc = np.load(data_path).astype(np.float32)

                    # Separate coordinates, features, and labels
                    coord = pc[:, :3]

                    # Extract features if any
                    feat = None
                    if pc.shape[1] > 3:
                        if self.label_field is not None:
                            # Last column is label, rest are features
                            if pc.shape[1] > 4:
                                feat = pc[:, 3:-1]
                            label = pc[:, -1:]
                        else:
                            # No labels, all non-xyz columns are features
                            feat = pc[:, 3:]
                            label = None
                    else:
                        label = None

                    # Apply voxel downsampling
                    if voxel_size:
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

                    # Reconstruct point cloud
                    if label is not None:
                        if feat is not None:
                            pc = np.hstack((coord, feat, label)).astype(np.float32)
                        else:
                            pc = np.hstack((coord, label)).astype(np.float32)
                    else:
                        if feat is not None:
                            pc = np.hstack((coord, feat)).astype(np.float32)
                        else:
                            pc = coord.astype(np.float32)

                    self.data.append(pc)

                npoints = np.array([len(data) for data in self.data])
                logging.info(
                    "split: %s, median npoints %.1f, avg num points %.1f, std %.1f"
                    % (
                        self.split,
                        np.median(npoints),
                        np.average(npoints),
                        np.std(npoints),
                    )
                )

                joblib.dump(self.data, processed_filename, compress=3)
                logging.info(f"{processed_filename} saved successfully")

            self.data_idx = np.arange(len(self.data_list))

        assert len(self.data_idx) > 0
        logging.info(f"Totally {len(self.data_idx)} samples in {split} set")

    def _tile_las_file(self, las_path: str, output_dir: str):
        """
        Tile a single LAS/LAZ file and save tiles as .npy files.

        Args:
            las_path: Path to LAS/LAZ file
            output_dir: Directory to save tiled .npy files
        """
        # Read LAS file
        las = laspy.read(las_path)

        # Extract coordinates (always XYZ)
        coords = np.vstack([las.x, las.y, las.z]).T
        coords -= coords.min(0)

        # Extract features
        features = []
        if self.use_approximate_hag:
            hag = compute_hag(coords, grid_size=5)
            features.append(hag.reshape(-1, 1))

        elif self.use_rgb:
            try:
                rgb = np.vstack([las.red, las.green, las.blue]).T
                # Normalize RGB to 0-1 if needed
                if rgb.max() > 255:
                    rgb = rgb / 65535.0
                else:
                    rgb = rgb / 255.0
                features.append(rgb)
            except AttributeError:
                logging.warning(f"RGB not found in {os.path.basename(las_path)}")

        if self.use_intensity:
            try:
                intensity = las.intensity.reshape(-1, 1)
                # Normalize intensity
                intensity = intensity / intensity.max()
                features.append(intensity)
            except AttributeError:
                logging.warning(f"Intensity not found in {os.path.basename(las_path)}")

        if self.use_return_number:
            try:
                return_num = las.return_number.reshape(-1, 1)
                num_returns = las.number_of_returns.reshape(-1, 1)
                features.append(return_num)
                features.append(num_returns)
            except AttributeError:
                logging.warning(
                    f"Return number not found in {os.path.basename(las_path)}"
                )

        # Extract labels if specified
        if self.label_field is not None:
            try:
                labels = getattr(las, self.label_field).reshape(-1, 1)
            except AttributeError:
                logging.warning(
                    f"Label field '{self.label_field}' not found in {os.path.basename(las_path)}, "
                    f"using zeros"
                )
                labels = np.zeros((len(las.x), 1))
        else:
            labels = None

        # Combine all data
        if features:
            feat_array = np.hstack(features)
            if labels is not None:
                pc = np.hstack([coords, feat_array, labels])
            else:
                pc = np.hstack([coords, feat_array])
        else:
            if labels is not None:
                pc = np.hstack([coords, labels])
            else:
                pc = coords
        tile_pc_fast(
            pc,
            las_path,
            output_dir,
            box_dim=self.tile_size,
            box_overlap=self.tile_overlap,
            voxel_max=self.voxel_max,
            min_points_per_tile=self.min_points_per_tile,
        )

    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns:
            dict: {'pos': coordinates, 'x': features, 'y': labels, 'heights': z-values}
        """
        data_idx = self.data_idx[idx % len(self.data_idx)]

        if self.presample:
            pc = self.data[data_idx]
            coord = pc[:, :3]
            coord -= np.min(coord, axis=0)
            # Parse features and labels based on configuration
            feat = None
            label = None

            if pc.shape[1] > 3:
                if self.label_field is not None:
                    if pc.shape[1] > 4:
                        feat = pc[:, 3:-1]
                    label = pc[:, -1:]
                else:
                    feat = pc[:, 3:]
        else:
            # Load from file
            data_path = self.data_list[data_idx]
            pc = np.load(data_path).astype(np.float32)

            # Extract coordinates
            coord = pc[:, :3]
            coord -= np.min(coord, axis=0)

            # Extract features and labels
            feat = None
            label = None

            if pc.shape[1] > 3:
                if self.label_field is not None:
                    # Last column is label
                    if pc.shape[1] > 4:
                        feat = pc[:, 3:-1]
                    label = pc[:, -1:]
                else:
                    # No labels, all are features
                    feat = pc[:, 3:]

            # Apply downsampling and cropping
            coord, feat, label = crop_pc(
                coord,
                feat,
                label,
                self.split,
                self.voxel_size,
                self.voxel_max,
                downsample=not self.presample,
                variable=self.variable,
                shuffle=self.shuffle,
            )

        # Prepare output
        if label is not None:
            label = label.squeeze(-1).astype(np.int64)
            # Apply label offset for remapping (e.g., [1,2,3] -> [0,1,2])
            if self.label_offset != 0:
                label = label + self.label_offset

        data = {"pos": coord, "x": feat, "y": label}

        # Apply transforms
        if self.transform is not None:
            data = self.transform(data)

        # Add heights if not already present
        if "heights" not in data.keys():
            data["heights"] = torch.from_numpy(
                coord[:, self.gravity_dim : self.gravity_dim + 1].astype(np.float32)
            )

        return data

    def __len__(self):
        return len(self.data_idx) * self.loop


if __name__ == "__main__":
    train_dataset = LASDataset(
        data_root="data/FORInstanceV2/",
        split="train",
        voxel_size=0.02,
        voxel_max=30000,
        label_field="semantic_seg",
        label_offset=-1,
        variable=True,
        presample=True,
        use_approximate_hag=True,
    )
    val_dataset = LASDataset(
        data_root="data/FORInstanceV2/",
        split="val",
        voxel_size=0.02,
        voxel_max=30000,
        label_field="semantic_seg",
        label_offset=-1,
        variable=True,
        presample=True,
        use_approximate_hag=True,
    )
    test_dataset = LASDataset(
        data_root="data/FORInstanceV2/",
        split="test",
        voxel_size=0.02,
        voxel_max=30000,
        label_field="semantic_seg",
        label_offset=-1,
        variable=True,
        presample=False,
        use_approximate_hag=True,
    )

    import pdb

    pdb.set_trace()
