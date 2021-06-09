from models import gat_layer
from data.mol_dataset import molset
from data.graph_utils import batch_graph, graph_collate, graph_inputs
from data.data_utils import split
from feat.atom_features import atom_features, one_of_k_encoding, one_of_k_encoding_unk
from models.gnn_base import GNN

import numpy as np
import pandas as pd
from pathlib import Path
from rdkit.Chem import MolFromSmiles
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

dataset_file = Path('datasets/tox21.csv.gz')

#tox21 = pd.read_csv(dataset_file)
targets = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',         
'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
tox21_molset = molset(dataset_file,'smiles', targets, normalise=False)
train, test = torch.utils.data.random_split(tox21_molset, split(tox21_molset, [.8, .2]),
                                            generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train, batch_size=100, collate_fn=graph_collate)

torch.cuda.set_device(0)

GAT_model = GNN(
    gat_layer,
    att_heads=2,
    input_features=75,
    hidden_channels=[300,300,300],
    activation='relu',
    dropout= 0.25,
    num_tasks=len(targets)
)

GAT_model.cuda()

optim = Adam(GAT_model.parameters(),lr=0.002)
bce = nn.BCELoss().cuda()

predictions = None

writer = SummaryWriter()
GAT_model.train()
GAT_model.init_weights()
epochs = 1000
for e in range(epochs):
    for b_i,batch in enumerate(train_loader):
        optim.zero_grad()
        predictions = GAT_model.forward(batch.X, batch.A, batch.graph_sizes)
        
        Y_b = torch.stack([torch.tensor(y) for y in batch.Y]).cuda()
        loss = bce(predictions,Y_b)#.cuda()
        loss.backward()
        optim.step()
        if (b_i%50==0):
            batch_num =  e*(len(train_loader.dataset) + b_i)
            writer.add_scalar('BCE loss: Train', loss.item(), batch_num)
            #binary_predictions = (predictions>0.5).float()
            #correct = (binary_predictions == y.float()).float().sum()
            print(f"Epoch {e}/{epochs}, Loss: {loss.item()}")#, Accuracy: {correct/len(y)}")
            for n,p in GAT_model.named_parameters():
                if 'bias' not in n:
                    writer.add_scalar(f'{n}:mean', p.mean().item(),batch_num)
                    writer.add_scalar(f'{n}:std', p.std().item(),batch_num)
writer.close()    