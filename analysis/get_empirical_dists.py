import numpy as np
from rdkit import Chem
from tqdm import tqdm
import os
from eval_bond_length import pair_distance_from_pos_v, bond_distance_from_mol
from eval_bond_length import get_pair_length_profile, get_bond_length_profile


if __name__ == '__main__':

    supplier = list(Chem.SDMolSupplier('/srv/ds/set-1/user/mahdi.ghorbani/FragDiff/datasets/geom_conformers.sdf'))

    all_pair_dists = []
    all_bond_dists = []
    for mol_id, mol in enumerate(supplier):
        try:
            pos = mol.GetConformer().GetPositions()

            atomicnums = []
            for atom in mol.GetAtoms():
                atomicnums.append(atom.GetAtomicNum())
            
            all_pair_dists += pair_distance_from_pos_v(pos, atomicnums)
            all_bond_dists += bond_distance_from_mol(mol)
        except:
            print(f'could not process mol {mol_id}')
    
    empirical_pair_length_profiles = get_pair_length_profile(all_pair_dists)
    empirical_bond_length_profiles = get_bond_length_profile(all_bond_dists)

    print(empirical_bond_length_profiles)
    print(empirical_pair_length_profiles)