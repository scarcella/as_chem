from pathlib import Path
import pandas as pd
from rdkit.Chem import MolFromSmiles
import numpy as np
import torch
from scipy.sparse import coo_matrix
import torch
import torch.sparse as sparse

def one_of_k_encoding(x, allowable_set):
  if x not in allowable_set:
    raise Exception("input {0} not in allowable set{1}:".format(
        x, allowable_set))
  return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
  """Maps inputs not in the allowable set to the last element."""
  if x not in allowable_set:
    x = allowable_set[-1]
  return list(map(lambda s: x == s, allowable_set))

def atom_features(atom,
                  bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=False):
  """Helper method used to compute per-atom feature vectors.

  Many different featurization methods compute per-atom features such as ConvMolFeaturizer, WeaveFeaturizer. This method computes such features.

  Parameters
  ----------
  bool_id_feat: bool, optional
    Return an array of unique identifiers corresponding to atom type.
  explicit_H: bool, optional
    If true, model hydrogens explicitly
  use_chirality: bool, optional
    If true, use chirality information.
  """
  if bool_id_feat:
    return np.array([atom_to_id(atom)])
  else:
    from rdkit import Chem
    results = one_of_k_encoding_unk(
      atom.GetSymbol(),
      [
          'C',
          'N',
          'O',
          'S',
          'F',
          'Si',
          'P',
          'Cl',
          'Br',
          'Mg',
          'Na',
          'Ca',
          'Fe',
          'As',
          'Al',
          'I',
          'B',
          'V',
          'K',
          'Tl',
          'Yb',
          'Sb',
          'Sn',
          'Ag',
          'Pd',
          'Co',
          'Se',
          'Ti',
          'Zn',
          'H',  # H?
          'Li',
          'Ge',
          'Cu',
          'Au',
          'Ni',
          'Cd',
          'In',
          'Mn',
          'Zr',
          'Cr',
          'Pt',
          'Hg',
          'Pb',
          'Unknown'
      ]) + one_of_k_encoding(atom.GetDegree(),
                             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
              ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
      results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                [0, 1, 2, 3, 4])
    if use_chirality:
      try:
        results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
      except:
        results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]

    return np.array(results)


class molecule(object):
    def __init__(self, data, smile_col, target_cols, norm=True):
        self.mol = self._smile2mol(data[smile_col])
        self.targets = self._extract_targets(data,target_cols)
        self.target_cols = target_cols
        self.mol_props = self._properties_matrix()    
        self.norm=norm
        if norm:
            self.adj_mat = self._normalise_adj()
        else:
            self.adj_mat = self._adj_mat()
    
        
    def _smile2mol(self, smile):
        return MolFromSmiles(smile)
    
    def _extract_targets(self, data, target_cols):
        targets ={}
        for t in target_cols:
            targets[t] = data[t]
        return targets
    
    def _properties_matrix(self):
        mol_props = [None]* self.mol.GetNumAtoms()
        for atom in self.mol.GetAtoms():
            mol_props[atom.GetIdx()] = atom_features(atom)
        return torch.tensor(mol_props, dtype=torch.float32, requires_grad=True)
    
    def _bond_index(self):
        mol = self.mol
        edges = self._self_bonds()
        for bond in iter(mol.GetBonds()):
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            edges.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        bond_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edges,bond_index
    
    def _adj_mat(self):
        _,b_i = self._bond_index()
        adj = torch.sparse_coo_tensor(b_i, torch.ones(b_i.shape[1]))
        return adj

    def _normalise_adj(self):
        adj_mat = self._adj_mat()
        degrees = sparse.sum(adj_mat,dim=1)
        d_ii = list(range(len(degrees)))
        D = torch.sparse_coo_tensor([d_ii,d_ii],degrees.values())
        D_inv_sqrt = D.pow(-0.5)
        norm_adj_mat = D_inv_sqrt.mm(adj_mat.to_dense()).mm(D_inv_sqrt.to_dense()) # need to get rid of dense conversions
        return norm_adj_mat    
    
    def _self_bonds(self):
        self_edges = [[i,i] for i in range(self.mol.GetNumAtoms())]
        return self_edges
    
    def get_feats(self):
        return self.mol_props
    
    def get_AM(self):
        return self.adj_mat
    
    def get_targets(self):
        return self.targets
    
    def get_model_inputs(self):
        return self.get_feats(), self.get_AM(), self.get_targets(), self.target_cols
    def __repr__(self):
        return self.mol.__repr__() # improve this
    
    def __str__(self):
        pass
    
