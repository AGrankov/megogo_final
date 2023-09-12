from torch.utils import data
import numpy as np

class MovieMLPDataset(data.Dataset):
    def __init__(self, wp_all, pairs=None):
        self.wp_all = wp_all
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs) if self.pairs is not None else len(self.wp_all)

    def __getitem__(self, index):
        if self.pairs is None:
            return self.wp_all[index].astype(np.float32)

        pair = self.pairs[index]
        uwp = self.wp_all[pair[0]]
        target_seq = pair[1]
        init_wp = uwp.copy()
        for idx, val in target_seq:
            init_wp[idx] -= val
        target_wp = ((uwp - init_wp) > 0.5).astype(np.float32)
        return init_wp.astype(np.float32), target_wp
