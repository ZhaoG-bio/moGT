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

import sys
import HGT_struction
from HGT_struction import HeteroDictLinear_drop,HGTConv

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    _, out = model(data.x_dict, data.edge_index_dict)
    mask = data['sample'].train_mask
    loss = F.cross_entropy(out[mask], data['sample'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)