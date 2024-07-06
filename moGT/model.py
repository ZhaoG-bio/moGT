from torch_geometric.data import HeteroData
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.datasets import DBLP
from torch_geometric.nn import Linear
import pickle
import torch.nn as nn
import torch_geometric.transforms as T
from collections import Counter
from sklearn import preprocessing
from torch_geometric.loader import NeighborLoader
from sklearn.model_selection import KFold
import copy




class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers,dropout = 0.2):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        self.dropout = dropout
        for node_type in datas.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels) 
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, datas.metadata(),
                           num_heads, dropout = self.dropout)  
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = F.leaky_relu(self.lin_dict[node_type](x), negative_slope=0.01) 

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            
        embedding_dict = x_dict

        return embedding_dict, self.lin(x_dict['sample'])