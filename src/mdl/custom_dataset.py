import torch
from torch.utils.data import Dataset
import scipy.sparse

class TFDataset(Dataset):
    def __init__(self, input_matrix, output_matrix):
        super().__init__()
        self.input = input_matrix
        self.output = output_matrix
    
    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, index):
        if scipy.sparse.issparse(self.input):
            return torch.as_tensor(self.input[index].toarray()).float(), torch.as_tensor(self.output[index].toarray())
        else:
            return torch.as_tensor(self.input[index]).float(), torch.as_tensor(self.output[index].toarray())