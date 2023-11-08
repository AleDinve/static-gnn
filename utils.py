import torch
from torch_geometric.nn import WLConv
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
from itertools import product
import numpy as np
import networkx as nx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WL(torch.nn.Module):
    def __init__(self,  num_it):
        super().__init__()
        self.num_it = num_it
        self.conv = WLConv()

    def forward(self, x, edge_index):
        for _ in range(self.num_it):
            x = self.conv(x, edge_index)   
        return x
def wl_colors(model, G, hashing = True):
    
    pred = model(G.x, G.edge_index)
    if hashing:
        pred = hash(tuple(sorted(pred.tolist())))
    else:
        pred, _, count = torch.unique(pred, return_inverse= True, return_counts = True)
        pred = torch.stack([pred,count])
    return pred
def wl_equiv_graphs():
    A1 = np.array([[0,1,1,0,0,0],[1,0,0,1,0,0],[1,0,0,1,1,0],
                    [0,1,1,0,0,1],[0,0,1,0,0,1],[0,0,0,1,1,0]])
    A2 = np.array([[0,1,1,0,0,0],[1,0,1,0,0,0],[1,1,0,1,0,0],
                    [0,0,1,0,1,1],[0,0,0,1,0,1],[0,0,0,1,1,0]])
    return A1, A2

def cycle_graph(n):

    A = np.zeros((n,n))
    A[0,-1]=1
    A[-1,0]=1
    for i in range(n-1):
        A[i,i+1]=1
        A[i+1,i]=1
    return A
    # s = np.shape(A)[0]
    # G = nx.from_numpy_matrix(A)
    # g = from_networkx(G)
    # g.x = torch.tensor([[1] for i in range(s)], dtype=torch.float)
    # g.y = torch.tensor([[1]],dtype=torch.float)
    # return g

def triangles():
    A = np.array([[0,1,1,0,0,0],[1,0,1,0,0,0],[1,1,0,0,0,0],
                 [0,0,0,0,1,1],[0,0,0,1,0,1],[0,0,0,1,1,0]])
    return A

def dyn_graphs1():
    num_it = 10
    num_nodes = 6
    A1, A2 = wl_equiv_graphs()
    # A1 = cycle_graph((6))
    # A2 = triangles()
    model = WL(num_it).to(device)
    model.eval()

    G1 = nx.from_numpy_matrix(A1).to_undirected()
    g1 = from_networkx(G1)
    g1.x = torch.tensor([[1] for i in range(num_nodes)], dtype=torch.float)
    g1.y = torch.tensor([[1]],  dtype=torch.float)
    G2 = nx.from_numpy_matrix(A2).to_undirected()
    g2 = from_networkx(G2)
    g2.x = torch.tensor([[1] for i in range(num_nodes)], dtype=torch.float)
    g2.y = torch.tensor([[1]],  dtype=torch.float)
    return g1,g2
    # print(dyn_wl(model, d1))
    # print(dyn_wl(model, d2))
    # print(dyn_wl(model, d3))
    # print(dyn_wl(model, d4))

def dyn_graphs2():
    num_it = 10
    num_nodes = 6
    # A1, A2 = wl_equiv_graphs()
    A1 = cycle_graph((6))
    A2 = triangles()
    model = WL(num_it).to(device)
    model.eval()
    G1 = nx.from_numpy_matrix(A1).to_undirected()
    g1 = from_networkx(G1)
    g1.x = torch.tensor([[1] for i in range(num_nodes)], dtype=torch.float)
    g1.y = torch.tensor([[0]],  dtype=torch.float)
    G2 = nx.from_numpy_matrix(A2).to_undirected()
    g2 = from_networkx(G2)
    g2.x = torch.tensor([[1] for i in range(num_nodes)], dtype=torch.float)
    g2.y = torch.tensor([[0]],  dtype=torch.float)
    return g1,g2

