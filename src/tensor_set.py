import torch
from torch.utils.data import Dataset


class TFDataset(Dataset):
    def __init__(self, input_matrix, output_matrix):
        super().__init__()
        self.input = input_matrix
        self.output = output_matrix

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, index):
        return torch.as_tensor(self.input[index].toarray()).float(), torch.as_tensor(self.output[index].toarray())