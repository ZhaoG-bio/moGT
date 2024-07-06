from torch_geometric.data import HeteroData
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import Linear
import pickle
import torch.nn as nn
import torch_geometric.transforms as T
from collections import Counter
import copy

import sys
import HGTConv
from HGTConv import HeteroDictLinear_drop,HGTConv

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    _, out = model(data.x_dict, data.edge_index_dict)
    mask = data['sample'].train_mask
    loss = F.cross_entropy(out[mask], data['sample'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)