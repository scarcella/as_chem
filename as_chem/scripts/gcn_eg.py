from data.as_mol_dataset import molset
from data.graph_utils import batch_graph, graph_collate, graph_inputs
from data.as_data_utils import split
from feat.as_atom_features import atom_features, one_of_k_encoding, one_of_k_encoding_unk
from models.as_gcn import AS_GCN

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
tox21_molset = molset(dataset_file,'smiles', ['SR-MMP'], normalise=False)
train, test = torch.utils.data.random_split(tox21_molset, split(tox21_molset, [.8, .2]),
                                            generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train, batch_size=100, collate_fn=graph_collate)

torch.cuda.set_device(0)

model = AS_GCN(75,[300,300,300],'relu',0.,1)

model.cuda()

optim = Adam(model.parameters(),lr=0.002)
bce = nn.BCELoss().cuda()

predictions = None
target = 'SR-MMP'
writer = SummaryWriter()
model.train()
model.init_weights()
epochs = 100
for e in range(epochs):
    for b_i,batch in enumerate(train_loader):
        optim.zero_grad()
        predictions = model.forward(batch.X, batch.A, batch.graph_sizes)
        y = torch.tensor([y[target] for y in batch.Y], device='cuda')
        loss = bce(predictions,y)#.cuda()
        loss.backward()
        optim.step()
        if (b_i%50==0):
            batch_num =  e*(len(train_loader.dataset) + b_i)
            writer.add_scalar('BCE loss: Train', loss.item(), batch_num)
            binary_predictions = (predictions>0.5).float()
            correct = (binary_predictions == y.float()).float().sum()
            print(f"Epoch {e}/{epochs}, Loss: {loss.item()}, Accuracy: {correct/len(y)}")
            for n,p in model.named_parameters():
                if 'bias' not in n:
                    writer.add_scalar(f'{n}:mean', p.mean().item(),batch_num)
                    writer.add_scalar(f'{n}:std', p.std().item(),batch_num)
writer.close()    