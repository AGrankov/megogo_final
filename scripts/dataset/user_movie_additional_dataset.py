from torch.utils import data
import numpy as np

class UserMovieAdditionalDataset(data.Dataset):
    def __init__(self,
                 users_movies_sequences,
                 additional_info,
                 movies_dict,
                 targets=None):
        self.users_movies_sequences = users_movies_sequences
        self.additional_info = additional_info
        self.movies_dict = movies_dict
        self.targets = targets

        self.data_size = len(list(self.movies_dict.values())[0])
        self.additional_size = 1 if len(additional_info.shape) < 3 else additional_info.shape[2]
        self.seq_len = len(users_movies_sequences[0])

    def __len__(self):
        return len(self.users_movies_sequences)

    def __getitem__(self, index):
        res_arr = np.zeros((self.seq_len, self.data_size + self.additional_size),
                            dtype=np.float32)

        for idx, seq_val in enumerate(self.users_movies_sequences[index]):
            if seq_val > 0:
                res_arr[idx, :self.data_size] = self.movies_dict[seq_val]

        res_arr[:, self.data_size:self.data_size+self.additional_size] = self.additional_info[index]

        if self.targets is not None:
            return res_arr, self.targets[index]

        return res_arr
