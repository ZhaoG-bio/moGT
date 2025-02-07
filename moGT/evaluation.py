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