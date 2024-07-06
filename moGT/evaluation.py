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

@torch.no_grad()
def test(model,data):
    model.eval()
    _, pred_scores = model(data.x_dict, data.edge_index_dict)
    pred = pred_scores.argmax(dim=-1)

    accs = []
    for split in ['train_mask', 'test_mask']:
        mask = data['sample'][split]
        acc = (pred[mask] == data['sample'].y[mask]).sum() / mask.sum()
        accs.append(float(acc))
    return accs