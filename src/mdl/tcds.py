import torch
from torch.utils.data import Dataset
import scipy.sparse

class TTFDataset(Dataset):
    def __init__(self, input_matrix, output_matrix, batch_size_list):
        super().__init__()
        self.input = input_matrix
        self.output = output_matrix
        self.batch_size_list = batch_size_list
    

    def __getitem__(self, index):
        start_idx = self.batch_size_list[index]
        end_idx = self.batch_size_list[index+1]
        input = self.input[start_idx:end_idx]
        output = self.output[start_idx:end_idx]
        if scipy.sparse.issparse(self.input):
            return torch.as_tensor(input.toarray()).float(), torch.as_tensor(output.toarray())
        else:
            return torch.as_tensor(input).float(), torch.as_tensor(output.toarray())
        
    def __len__(self):
        return len(self.batch_size_list) - 1