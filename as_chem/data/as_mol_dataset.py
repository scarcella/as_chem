import pandas as pd
import torch

from feat.as_mol import molecule
from torch.utils.data import Dataset, DataLoader

class molset(Dataset):
    def __init__(self,
                filename,
                smile_col,
                target_cols,
                normalise=False
                ):
        super(molset).__init__()
        self.df =  pd.read_csv(filename).dropna(subset=target_cols).reset_index()
        self.mols = self._smile2mol(smile_col, target_cols)
        if normalise:
            self.x_mu, self.x_std = self.std_mean()
            self.normalise_inputs()
        
    def _smile2mol(self, smile_col, target_cols):
        mols = {}
        for i, row in self.df.iterrows():
                mols[i] = molecule(row, smile_col, target_cols)
        
        return mols
     
    def std_mean(self):
        x_ =[self.mols[i].get_feats() for i in range(len(self.mols))]
        x = torch.cat(x_,0)
        x_mu, x_std = torch.std_mean(x,dim=0)
        x_std[x_std==0]=1 # as per https://github.com/scikit-learn/scikit-learn/blob/7389dbac82d362f296dc2746f10e43ffa1615660/sklearn/preprocessing/data.py#L70
        return  x_mu, x_std
        
    def normalise_inputs(self):
        for i in range(len(self.mols)):
            x_norm = self.mols[i].get_feats().data.sub_(self.x_mu[None, :]).div_(self.x_std[None, :])
            self.mols[i].set_feats(x_norm)

    def __getitem__(self, idx):
        return self.mols[idx].get_model_inputs()
    
    def __len__(self):
        return len(self.df)
    