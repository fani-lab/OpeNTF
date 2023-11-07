import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.data import HeteroData
from tqdm import tqdm

# import classes from opentf
import param
import sys,os, json
from json import JSONEncoder
from cmn.team import Team
from cmn.author import Author
from cmn.publication import Publication

'''
Class Definitions here
'''
# GCN class to form a GCN model
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

'''class definition ends here'''

# a method to create a custom dataset
def create_data(x, edge_index, y = None):
    # x = torch.tensor([-1, 0, 1])
    # edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    # y = torch.tensor([2, 4, 1])

    # the data is only for generating the embedding without any target labels
    if not y :
        data = Data(x = x, edge_index = edge_index)
    else:
        data = Data(x = x, edge_index = edge_index, y = y)
    return data

def load_dataset():
    # Load a graph dataset
    # dataset = KarateClub()
    dataset = Planetoid(root = '~/tmp/Cora', name = 'Cora')


    return dataset

def explore_dataset(dataset):
    data = dataset[0]
    print(f'Dataset: {dataset}:')
    print('======================')

    print(f'dimension of the data : {len(data)}')

    # print(f'data.x = {data.x}')
    # print(f'data.edge_index = {data.edge_index}')

    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    # print(f'Range of training nodes : {data.train_mask}')
    print(f'Type of data : {type(data)}')
    print(f'')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
    print(f'Contains self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

# explore the particular data points in the dataset
def explore_dataset_specific(dataset):
    data = dataset[0]

    print(f'###############################################')
    print(f'explore_dataset_specific')
    print(f'###############################################')
    print()

    # size of a tensor object
    torch.set_printoptions(threshold=10_000)
    print(f'type of data object : {type(data)}')
    print(f'Size of data : {data.size()}')
    print(f'Type of the data.x : {type(data.x)}')
    print(f'Size of x = {data.x.size()}')
    print(f'Size of y = {data.y.size()}')
    print(f'Size of edge_index : {data.edge_index.size()}')
    print(f'Number of nodes in x and y = {data.x.size()[0]}')
    print(f'Number of features in each node = {data.x.size()[1]}')
    print()
    print(f'x :')
    print('-------------------')
    print(data.x)
    print(f'y : ')
    print(f'------------------')
    print(data.y)
    print(f'edge_index : ')
    print(f'------------------')
    print(data.edge_index)
    print(f'###############################################')
    print()

# visualize the data
def visualize_dataset(data):
    # G = to_networkx(data, to_undirected=True)
    # nx.draw(G)
    pass

def create_model(dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device = {device}')
    model = GCN(dataset).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 5e4)
    return model, data, optimizer
    pass

def train_model(model, data, optimizer):
    model.train()
    for epoch in tqdm(range(200)):
        print(f'Epoch : {epoch}')
        optimizer.zero_grad()
        out = model(data)
        # print(f'output after each pass : {out}')
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        print(f'Loss before optimization : {loss}')
        loss.backward()
        optimizer.step()
        print(f'loss after backprop {epoch} : {loss}')
    return model, data


def evaluate_model(model, data):
    model.eval()
    pred = model(data).argmax(dim = 1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')



# preprocess a data into graph data
def preprocess(dataset):
    print()
    print(f'dataset = {dataset}')

    assert len(indexes['i2s']) == len(indexes['s2i'])
    assert len(indexes['i2t']) == len(indexes['t2i'])
    assert len(indexes['i2c']) == len(indexes['c2i'])
    assert len(teams) == len(indexes['i2t'])

    num_teams = len(indexes['i2t'])
    num_skills = len(indexes['i2s'])
    num_experts = len(indexes['i2c'])
    num_edges_t2s = 0
    num_edges_t2e = 0


    # dataset now contains indexes and teams
    # indexes is the index containing all the distinct candidates, skills and teams info
    # teams is a dict containing a Team object against each key (1,2,3.....nteams)
    # each Team object holds the info of distinct teams (e.g : distinct publications)

    # we build a graph where
    # num_nodes = num of distinct candidate + skill + team
    # edge exists between each team node to its relevant candidate and skill node

    # the graph team_graph is basically a heterodata with node types team t, skill s and experts e.
    # only the team t type nodes will contain the success or failure as y, all other properties like
    # x, edge_index and edge_features will be the same

    ### Nodes
    # x = [num_teams, num_features_teams], here,
    # num_features_teams = the number of features each team might contain
    team_graph['team'].x = torch.tensor((num_teams, 1), dtype = None)
    # x = [num_skills, num_features_skills]
    team_graph['skill'].x = torch.tensor(list(indexes['i2s'].keys()))
    # x = [num_experts, num_features_experts]
    team_graph['expert'].x = torch.tensor(list(indexes['i2c'].keys()))

    ### Edges
    # so here the edges are mentioned as source_node "does_something_to" target_node
    # which means, in our case, each team contains a skill or expert
    # "team has skills" / t2s and "team contains experts" / t2e
    # specifically, "publication has fos" or "publication contains authors"
    # so we have 2 types of edges t->s and t->e

    # edge_index = [2, num_edge_t2s], the edge_index for the type "has"
    team_graph['team', 'has', 'skill'].edge_index = ...
    # edge_index = [2, num_edge_t2e], the edge_index for the type "contains"
    team_graph['team', 'contains', 'expert'].edge_index = ...



    # create nodes
    print('\nTeams : ')
    for i, team_index in indexes['i2t'].items():
        print(f'i = {i}, team = {team_index} \n\tteam_id : {teams[team_index].id} \n\tteam_title : {teams[team_index].title} \n\tmember[0].name : {teams[team_index].members[0].name}')
        # for each team, we connect each t with each t.fos




def raw_gcn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # create a sample data to test
    # this data will have 3 nodes 0, 1 and 2 with 0-1, 1-2 but no 0-2 edge
    # the similarity should be between 0-1, 1-2 but 0-2 should be different from one another
    data = create_data(torch.tensor([[0], [1], [2]], dtype=torch.float), torch.tensor([[0, 1, 1, 2],[1, 0, 2, 1]], dtype=torch.long)).to(device)
    model = GCN(input_dim = data.num_node_features, hidden_dim = 16, output_dim = 3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')


# download the dataset

def movie_interactions():
    print('####################################')
    print('Movie Interactions Model')
    print('####################################')


def main():
    dataset = Planetoid(root='/tmp/Cora', name='Cora')

if __name__ == "__main__":
    # dataset = Planetoid(root='/tmp/Cora', name='Cora')

    # cls = Publication
    # domain = 'dblp'
    # filename = 'toy.dblp.v12.json'
    # datapath = 'data/raw/dblp/toy.dblp.v12.json'
    # output = 'data/preprocessed/' + domain + '/' + filename + '/gnn'
    # # create the empty graph for all the teams
    # team_graph = HeteroData()
    #
    # # read the data based on the domain
    # # 'indexes' contains the list of all the distinct
    # # candidates, skills and teams
    # # teams is a dict with keys 1,2,3, ..... nteams. against each key there is a Team object holding a team
    # # so info of each team can be accessed with teams[3].title, teams[3].fos etc. (here 3 is the key of the team)
    # indexes, teams = cls.read_data(datapath, output, False, False, param.settings['data'])
    # dataset = {'index': indexes, 'data' : teams}
    # preprocess(dataset)


    # main()
    raw_gcn()
    # planetoid()
    # movie_interactions()


