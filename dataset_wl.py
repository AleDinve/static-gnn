'''@author: Giuseppe Alessio D'Inverno'''
'''@date: 02/03/2023'''

import torch
from torch_geometric.nn import WLConv
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
from itertools import product
import os.path as osp
import os
from utils import *
from torch_geometric.datasets import TUDataset
import numpy as np
import networkx as nx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class QM9_reduct(InMemoryDataset):
    def __init__(self, root, num_it, batch_size, threshold,
                 transform=None, pre_transform=None, pre_filter=None):
        self.num_it = num_it
        self.batch_size = batch_size
        self.threshold = threshold
        super().__init__(root, transform, pre_transform, pre_filter)  
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list, _ = dataset_gen(self.num_it, 'QM9', self.batch_size, self.threshold)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0]) 

def dataset_subsample(dataset_list, color_list, thresh):
    color_unique, inv, count = np.unique(color_list, return_inverse = True, return_counts=True)
    count_ext = count[inv]
    color_list = (np.array(color_list)[count_ext>thresh]).tolist()
    subsampled = dataset_list[count_ext>thresh]
    print(len(subsampled))
    return subsampled, color_list

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

def dataset_retrieve(dataset):
    
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data_new', 'TU')
    if not osp.exists(path):
        os.makedirs(path)
    dataset_list = TUDataset(path, name=dataset).shuffle()
    return dataset_list, dataset_list.num_features


def dataset_gen(num_it, dataset_name, batch_size, threshold):
    model = WL(num_it).to(device)
    model.eval()
    dataset_list, num_features = dataset_retrieve(dataset_name)
    print(dataset_list.len())
    color_list = []
    dataset_copy = []
    for i, data in enumerate(dataset_list): 
        data.x = torch.tensor([[1] for j in range(data.num_nodes)])
        dataset_copy.append(data)
        col = wl_colors(model, data, hashing = True)
        color_list.append(col)
    #subsample datasets retaining wl classes with more than 45 elements
    dataset_sub, color_list = dataset_subsample(dataset_list, color_list, threshold)                                        
    ord_color_list, ind_inverse, count_color = np.unique(color_list, return_counts=True,
                                             return_inverse=True)
    natural_list = np.array([j for j in range(len(ord_color_list))])
    color_list = natural_list[ind_inverse].tolist()
    amin, amax = min(color_list), max(color_list)
    for i, val in enumerate(color_list):
        color_list[i] = (val-amin) / (amax-amin) 
    sub_copy = []
    for i, g  in enumerate(dataset_sub):
        g.x = torch.tensor([[1] for j in range(g.num_nodes)],  dtype=torch.float)
        g.y = torch.tensor([[color_list[i]]],  dtype=torch.float)
        sub_copy.append(g)
    num_classes = np.size(natural_list)
    return sub_copy, num_classes

def dataloader_gen(dataset_name, num_it, batch_size, threshold):
    dataset = QM9_reduct(dataset_name, num_it, batch_size, threshold)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_features = 1
    return loader, num_features, dataset.num_classes
        
def trial_loader(batch_size):
    dataset = []
    for i in range(12):
        dataset += dyn_graphs1()
        dataset += dyn_graphs2()
    loader = DataLoader(dataset, batch_size, shuffle=True)
    return loader


    
if __name__ == '__main__':
    datasets = ["ENZYMES","PROTEINS", "NCI1", "PTC_MR","IMDB-BINARY","QM9"]
    dataset_name = datasets[-1]
    batch_size = 16
    dataset_gen(12, dataset_name, batch_size)