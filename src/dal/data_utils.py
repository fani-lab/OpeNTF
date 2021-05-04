import json
from scipy.sparse import csr_matrix, lil_matrix
import torch
import numpy as np
from cmn.member import Member
from cmn.author import Author
from cmn.team import Team
from cmn.document import Document
from datetime import datetime

def read_data(data_path):
    counter = 0
    all_docs = {}
    all_authors = {}  
    input_data = []
    output_data = []

    with open(data_path, "r") as jf:
        # Skip the first line
        jf.readline() 
        while counter < 1000:
            # Read line by line to not overload the memory
            line = jf.readline().lower().lstrip(",")
            jsonline = json.loads(line)

            # Retrieve the desired attributes
            doc_id = jsonline['id']
            doc_title = jsonline['title']
            doc_year = jsonline['year']
            doc_type = jsonline['doc_type']
            doc_venue = jsonline['venue']
            
            if 'references' in jsonline.keys():
                doc_references = jsonline['references']
            else:
                doc_references = []

            if 'fos' in jsonline.keys():    
                doc_fos = jsonline['fos']
            else:
                continue
            if 'keywords' not in jsonline.keys():
                doc_keywords = []
            else:
                doc_keywords = jsonline['keywords']
                
            authors = []
            for auth in jsonline['authors']:
                
                # Retrieve the desired attributes
                auth_id = auth['id']
                auth_name = auth['name']
                
                if 'org' in auth.keys():
                    auth_org = auth['org']
                else:
                    auth_org = ""
                
                if auth_id not in all_authors.keys():
                    author = Author(auth_id, auth_name, auth_org)
                    all_authors[auth_id] = author
                else:
                    author = all_authors[auth_id]
                authors.append(author) 

                # author = Author(auth_id, auth_name, auth_org)
                # authors.append(author)
                # if auth_id not in all_authors.keys():
                #     all_authors[auth_id] = author
                    
            doc = Document(doc_id, authors, doc_title, doc_year,doc_type, doc_venue, doc_references, doc_fos, doc_keywords)
            if doc.get_uid() not in all_docs.keys():
                all_docs[doc.get_uid()] = doc

            input_data.append(doc.get_fields())
            output_data.append(doc.get_members_names())

            counter += 1

    return all_authors, all_docs, input_data, output_data

def build_index_authors(all_authors):
    idx = 0
    author_to_index = {}
    index_to_author = {}

    for auth in all_authors.values():
        index_to_author[idx] = auth.get_id()
        author_to_index[auth.get_id()] = idx
        idx += 1

    return index_to_author, author_to_index

def build_index_skills(all_docs):
    idx = 0
    skill_to_index = {}
    index_to_skill = {}

    for doc in all_docs.values():
        for field in doc.get_fields():
            if field not in skill_to_index.keys():
                skill_to_index[field] = idx
                index_to_skill[idx] = field
                idx += 1

    return index_to_skill, skill_to_index


def build_dataset(all_docs, skill_to_index, author_to_index):

    training_size = len(all_docs)
    BATCH_SIZE = 100
    SKILL_SIZE = len(skill_to_index)
    AUTHOR_SIZE = len(author_to_index)

    #Sparse Matrix and bucketing
    data = lil_matrix((training_size, SKILL_SIZE + AUTHOR_SIZE + 1))
    data_ = np.zeros((BATCH_SIZE, SKILL_SIZE + AUTHOR_SIZE + 1))
    j = -1
    for i, doc in enumerate(all_docs.values()):
        if i >= training_size: break

        # Generating one hot encoded vector for input
        X = np.zeros((1, SKILL_SIZE))
        input_fields = doc.get_fields()
        for field in input_fields:
            X[0, skill_to_index[field]] = 1

        # This does not work since the number of authors are different for each sample, therefore we need to build the output as a one hot encoding
        # y_index = []
        # for id in output_ids:
        #     y_index.append(author_to_index[id])
        # y_index.append(len(output_ids))
        # y = np.asarray([y_index])

        # Generating one hot encoded vector for output
        y = np.zeros((1, AUTHOR_SIZE + 1))
        output_ids = doc.get_members_ids()
        for id in output_ids:
            y[0, author_to_index[id]] = 1
        y[0, -1] = len(output_ids)
        

        # Building a training instance
        X_y = np.hstack([X, y])

        # Bucketing
        try:
            j += 1
            data_[j] = X_y
        except:
            s = int(((i / BATCH_SIZE) - 1) * BATCH_SIZE)
            e = int(s + BATCH_SIZE)
            data[s: e] = data_
            j = 0
            data_[j] = X_y
        if (i % BATCH_SIZE == 0):
            print(f'Loading {i}/{len(all_docs)} instances!{datetime.now()}')
    if j > -1:
        data[-j-1:] = data_[0:j+1]

    
    input_matrix = data[:, :SKILL_SIZE]
    output_matrix = data[:, -1-AUTHOR_SIZE:]
    print(input_matrix.shape)
    print(output_matrix.shape)
    # input_matrix = torch.rand(100,100)
    # output_matrix = torch.rand(100,100)


    return input_matrix, output_matrix


# def build_dataset(all_docs, skill_to_index, author_to_index):
    
#     counter = 0

#     input_data = []
#     skill_row = []
#     skill_col = []

#     output_data = []
#     auth_row = []
#     auth_col = []
    
#     for doc in all_docs.values():

#         inp_fields = doc.get_fields()
#         for field in inp_fields:
#             skill_row.append(counter)
#             skill_col.append(skill_to_index[field])
#             input_data.append(1)

#         out_ids = doc.get_members_ids()
#         for id in out_ids:
#             auth_row.append(counter)
#             auth_col.append(author_to_index[id])
#             output_data.append(1)
        
#         auth_row.append(counter)
#         auth_col.append(len(author_to_index))
#         output_data.append(len(out_ids))

#         counter += 1

    
#     input_matrix = csr_matrix((input_data, (skill_row, skill_col)), shape=(len(all_docs), len(skill_to_index)))
#     output_matrix = csr_matrix((output_data, (auth_row, auth_col)), shape=(len(all_docs), len(author_to_index)+1))

#     # input_matrix = torch.sparse.LongTensor(torch.LongTensor([input_matrix.tocoo().row.tolist(), input_matrix.tocoo().col.tolist()]), torch.LongTensor(input_matrix.tocoo().data.astype(np.int32)))
#     # output_matrix = torch.sparse.LongTensor(torch.LongTensor([output_matrix.tocoo().row.tolist(), output_matrix.tocoo().col.tolist()]), torch.LongTensor(output_matrix.tocoo().data.astype(np.int32)))

#     return input_matrix, output_matrix


# def build_dataset(all_docs, skill_to_index, author_to_index):
    
#     counter = 0

#     input_data = []
#     skill_row = []
#     skill_col = []

#     output_data = []
#     auth_row = []
#     auth_col = []
    
#     for doc in all_docs.values():

#         inp_fields = doc.get_fields()
#         for field in inp_fields:
#             skill_row.append(counter)
#             skill_col.append(skill_to_index[field])
#             input_data.append(1)

#         out_ids = doc.get_members_ids()
#         for id in out_ids:
#             auth_row.append(counter)
#             auth_col.append(author_to_index[id])
#             output_data.append(1)
        
#         auth_row.append(counter)
#         auth_col.append(len(author_to_index))
#         output_data.append(len(out_ids))

#         counter += 1

#     # skill_row_tensor = torch.tensor(skill_row)
#     # skill_col_tensor = torch.tensor(skill_col)
#     # auth_row_tensor = torch.tensor(auth_row)
#     # auth_col_tensor = torch.tensor(auth_col)
#     # input_data_tensor = torch.tensor(input_data)
#     # output_data_tensor = torch.tensor(output_data)

#     input_matrix = csr_matrix((input_data, (skill_row, skill_col)), shape=(len(all_docs), len(skill_to_index)))
#     output_matrix = csr_matrix((output_data, (auth_row, auth_col)), shape=(len(all_docs), len(author_to_index)+1))
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