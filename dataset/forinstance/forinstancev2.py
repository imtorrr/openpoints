from ..build import DATASETS
from ..las_dataset import LASDataset
import numpy as np


@DATASETS.register_module()
class FORInstanceV2(LASDataset):
    classes = ["ground", "stem", "vegetation"]

    num_classes = 3
    num_per_class = np.array([9078718, 8379647, 41767928], dtype=np.int32)
    class2color = {
        "ground": [0, 0, 255],
        "stem": [0, 255, 0],
        "vegetation": [255, 0, 0],
    }
    cmap = [*class2color.values()]
    gravity_dim = 2

    def __init__(self, *args, max_samples=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Limit dataset size for testing
        if max_samples is not None:
            np.random.seed(0)
            self.data_idx = np.random.choice(self.data_idx, size=max_samples)
