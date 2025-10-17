"""
Author: PointNeXt
"""

import numpy as np
import torch
from easydict import EasyDict as edict
from openpoints.utils import registry
from openpoints.transforms import build_transforms_from_cfg

DATASETS = registry.Registry("dataset")


def concat_collate_fn(datas):
    """collate fn for point transformer"""
    pts, feats, labels, offset, count, batches = [], [], [], [], 0, []
    for i, data in enumerate(datas):
        count += len(data["pos"])
        offset.append(count)
        pts.append(data["pos"])
        feats.append(data["x"])
        labels.append(data["y"])
        batches += [i] * len(data["pos"])

    data = {
        "pos": torch.cat(pts),
        "x": torch.cat(feats),
        "y": torch.cat(labels),
        "o": torch.IntTensor(offset),
        "batch": torch.LongTensor(batches),
    }
    return data


def collate_to_pyg_batch(batch):
    """
    Collate function to combine a list of point cloud samples into a single
    `torch_geometric.data.Batch` for use with standard `torch.utils.data.DataLoader`.

    This function is typically used in point cloud segmentation tasks (e.g., Point Transformer),
    where each item in the dataset returns a dictionary containing:
      - `"pos"`: (N, 3) float tensor of point coordinates.
      - `"x"`:   (N, C) optional feature tensor or `None`.
      - `"y"`:   (N,)   long tensor of labels.
      - (optional) `"heights"` or other per-point attributes.

    Args:
        batch (List[Dict[str, Any]]): A list of dataset items, each a dict with keys
            compatible with `torch_geometric.data.Data`.

    Returns:
        Batch: A `torch_geometric.data.Batch` object containing all samples concatenated
        along the point dimension, ready to be used in training or inference.

    Example:
        >>> loader = DataLoader(dataset, batch_size=8, collate_fn=concat_collate_fn)
        >>> for batch in loader:
        ...     print(batch.pos.shape)  # -> (total_points_in_batch, 3)
    """
    from torch_geometric.data import Data, Batch

    data_list = [Data(**item) for item in batch]
    return Batch.from_data_list(data_list)


def build_dataset_from_cfg(cfg, default_args=None):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        cfg (eDICT):
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    return DATASETS.build(cfg, default_args=default_args)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataloader_from_cfg(
    batch_size,
    dataset_cfg=None,
    dataloader_cfg=None,
    datatransforms_cfg=None,
    split="train",
    distributed=True,
    dataset=None,
):
    if dataset is None:
        if datatransforms_cfg is not None:
            # in case only val or test transforms are provided.
            if split not in datatransforms_cfg.keys() and split in ["val", "test"]:
                trans_split = "val"
            else:
                trans_split = split
            data_transform = build_transforms_from_cfg(trans_split, datatransforms_cfg)
        else:
            data_transform = None

        if split not in dataset_cfg.keys() and split in ["val", "test"]:
            dataset_split = "test" if split == "val" else "val"
        else:
            dataset_split = split
        split_cfg = dataset_cfg.get(dataset_split, edict())
        if split_cfg.get("split", None) is None:  # add 'split' in dataset_split_cfg
            split_cfg.split = split
        split_cfg.transform = data_transform
        dataset = build_dataset_from_cfg(dataset_cfg.common, split_cfg)

    collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
    collate_fn = (
        dataloader_cfg.collate_fn
        if dataloader_cfg.get("collate_fn", None) is not None
        else collate_fn
    )
    collate_fn = eval(collate_fn) if isinstance(collate_fn, str) else collate_fn

    shuffle = split == "train"
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=int(dataloader_cfg.num_workers),
            worker_init_fn=worker_init_fn,
            drop_last=split == "train",
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=int(dataloader_cfg.num_workers),
            worker_init_fn=worker_init_fn,
            drop_last=split == "train",
            shuffle=shuffle,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    return dataloader
