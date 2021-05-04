import json
from scipy.sparse import csr_matrix, lil_matrix
import torch
import numpy as np
from cmn.member import Member
from cmn.author import Author
from cmn.team import Team
from cmn.document import Document
from datetime import datetime




# def build_dataset(teams, skill_to_index, member_to_index):
    
#     counter = 0

#     input_data = []
#     skill_row = []
#     skill_col = []

#     output_data = []
#     auth_row = []
#     auth_col = []
    
#     for doc in teams.values():

#         inp_fields = doc.get_fields()
#         for field in inp_fields:
#             skill_row.append(counter)
#             skill_col.append(skill_to_index[field])
#             input_data.append(1)

#         out_ids = doc.get_members_ids()
#         for id in out_ids:
#             auth_row.append(counter)
#             auth_col.append(member_to_index[id])
#             output_data.append(1)
        
#         auth_row.append(counter)
#         auth_col.append(len(member_to_index))
#         output_data.append(len(out_ids))

#         counter += 1

    
#     input_matrix = csr_matrix((input_data, (skill_row, skill_col)), shape=(len(teams), len(skill_to_index)))
#     output_matrix = csr_matrix((output_data, (auth_row, auth_col)), shape=(len(teams), len(member_to_index)+1))

#     # input_matrix = torch.sparse.LongTensor(torch.LongTensor([input_matrix.tocoo().row.tolist(), input_matrix.tocoo().col.tolist()]), torch.LongTensor(input_matrix.tocoo().data.astype(np.int32)))
#     # output_matrix = torch.sparse.LongTensor(torch.LongTensor([output_matrix.tocoo().row.tolist(), output_matrix.tocoo().col.tolist()]), torch.LongTensor(output_matrix.tocoo().data.astype(np.int32)))

#     return input_matrix, output_matrix


# def build_dataset(teams, skill_to_index, member_to_index):
    
#     counter = 0

#     input_data = []
#     skill_row = []
#     skill_col = []

#     output_data = []
#     auth_row = []
#     auth_col = []
    
#     for doc in teams.values():

#         inp_fields = doc.get_fields()
#         for field in inp_fields:
#             skill_row.append(counter)
#             skill_col.append(skill_to_index[field])
#             input_data.append(1)

#         out_ids = doc.get_members_ids()
#         for id in out_ids:
#             auth_row.append(counter)
#             auth_col.append(member_to_index[id])
#             output_data.append(1)
        
#         auth_row.append(counter)
#         auth_col.append(len(member_to_index))
#         output_data.append(len(out_ids))

#         counter += 1

#     # skill_row_tensor = torch.tensor(skill_row)
#     # skill_col_tensor = torch.tensor(skill_col)
#     # auth_row_tensor = torch.tensor(auth_row)
#     # auth_col_tensor = torch.tensor(auth_col)
#     # input_data_tensor = torch.tensor(input_data)
#     # output_data_tensor = torch.tensor(output_data)

#     input_matrix = csr_matrix((input_data, (skill_row, skill_col)), shape=(len(teams), len(skill_to_index)))
#     output_matrix = csr_matrix((output_data, (auth_row, auth_col)), shape=(len(teams), len(member_to_index)+1))
#     # input_matrix = torch.sparse_csr_tensor(skill_row_tensor, skill_col_tensor, input_data_tensor)
#     # output_matrix = torch.sparse_csr_tensor(auth_row_tensor, auth_col_tensor, output_data_tensor)
#     # input_matrix = torch.sparse_coo_tensor([skill_row_tensor, skill_col_tensor], input_data_tensor)
#     # output_matrix = torch.sparse_coo_tensor([auth_row_tensor, auth_col_tensor], output_data_tensor)
#     # input_matrix = torch.sparse_coo_tensor([skill_row, skill_col], input_data)
#     # output_matrix = torch.sparse_coo_tensor([auth_row, auth_col], output_data)
    
    
#     return input_matrix, output_matrix


# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct/num_samples