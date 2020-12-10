

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