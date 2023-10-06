from scipy import spatial as sci_spatial
import numpy as np
from collections import Counter


CROSSDOCK_atom_charges =  {'C':6, 'N': 7, 'O': 8, 'S': 16, 'B': 5, 'Br': 35, 'Cl': 17, 'P': 15, 'I':53 ,'F':9}

def get_atom_charges(mol, charge_dict):
    atomic_nums = []
    for atom in mol.GetAtoms():
        atomic_nums.append(charge_dict[atom.GetSymbol()])
    
    atomic_nums = np.array(atomic_nums)
    return atomic_nums

ATOM_TYPE_DISTRIBUTION = { # atom type distributions in CrossDock
    6: 0.6715020339893559,
    7: 0.11703509510732567,
    8: 0.16956379168491933,
    9: 0.01307879304486639,
    15: 0.01113716146426898,
    16: 0.01123926340861198,
    17: 0.006443861300651673,
}

ATOM_TYPE_DISTRIBUTION_GEOM = { # atom type distributions in CrossDock
    6: 0.7266496963585743,
    7: 0.11690156566351215,
    8: 0.11619156632264795,
    9: 0.008849559988534103,
    15: 0.0001854777473386173,
    16: 0.022003011957949646,
    17: 0.007286864677748788,
    35: 0.001897001182960629,
}

def eval_atom_type_distribution(pred_counter: Counter, data_type='GEOM'):
    total_num_atoms = sum(pred_counter.values())
    pred_atom_distribution = {}
    if data_type == 'GEOM':
        for k in ATOM_TYPE_DISTRIBUTION_GEOM:
            pred_atom_distribution[k] = pred_counter[k] / total_num_atoms
        js = sci_spatial.distance.jensenshannon(np.array(list(ATOM_TYPE_DISTRIBUTION_GEOM.values())),
                                                np.array(list(pred_atom_distribution.values())))
    elif data_type == 'CrossDock':
        for k in ATOM_TYPE_DISTRIBUTION:
            pred_atom_distribution[k] = pred_counter[k] / total_num_atoms
        js = sci_spatial.distance.jensenshannon(np.array(list(ATOM_TYPE_DISTRIBUTION.values())),
                                                np.array(list(pred_atom_distribution.values())))    
    
    return js