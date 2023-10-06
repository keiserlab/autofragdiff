from pathlib import Path
from time import time
import argparse
import shutil
import random 
import matplotlib.pyplot as plt

from tqdm import tqdm 
import numpy as np

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa, three_to_one
from rdkit import Chem

amino_acid_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
pocket_atom_dict =  {'C': 0, 'N': 1, 'O': 2, 'S': 3} # only 4 atoms types for pocket
atom_dict = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9}


def get_one_hot(atom, atoms_dict):
    one_hot = np.zeros(len(atoms_dict))
    one_hot[atoms_dict[atom]] = 1
    return one_hot

def process_pocket(pdbfile, sdffile, atom_dict, pocket_atom_dict, dist_cutoff, remove_H=True, ca_only=False):

    pdb_struct = PDBParser(QUIET=True).get_structure('', pdbfile)

    try:
        ligand = Chem.SDMolSupplier(str(sdffile))[0]
    except:
        raise Exception(f'cannot read sdf mol ({sdffile})')
    
    # remove H atom if not in atom_dict, other atom types taht aren't allowed
    # should stay so that the entire ligand can be removed from the dataset
    lig_atoms = [a.GetSymbol() for a in ligand.GetAtoms()
                 if (a.GetSymbol().capitalize() in atom_dict or a.element !='H')]
    lig_coords = np.array([list(ligand.GetConformer(0).GetAtomPosition(idx))
                           for idx in range(ligand.GetNumAtoms())])

    # find interacting pocket residues based on distance cutoff
    pocket_residues = []
    for residue in pdb_struct[0].get_residues():
        res_coords = np.array([a.get_coord() for a in residue.get_atoms()])
        if is_aa(residue.get_resname(), standard=True) and \
                (((res_coords[:, None, :] - lig_coords[None, :, :]) ** 2).sum(-1)**0.5).min() < dist_cutoff:
            pocket_residues.append(residue)

    
    pocket_ids = [f'{res.parent.id}:{res.id[1]}' for res in pocket_residues]
        
    if ca_only:
        try:
            pocket_one_hot = []
            pocket_coords = []
            for res in pocket_residues:
                for atom in res.get_atoms():
                    if atom.name == 'CA':
                        pocket_one_hot.append(np.eye(1, len(amino_acid_dict),
                        amino_acid_dict[three_to_one(res.get_resname())]).squeeze())
                    pocket_coords.append(atom.coord)
            pocket_one_hot = np.stack(pocket_one_hot)
            pocket_coords = np.stack(pocket_coords)
        except KeyError as e:
            raise KeyError(f'{e} not in amino acid dict ({pdbfile}, {sdffile})')
    else: 
        full_atoms = np.concatenate([np.array([atom.element for atom in res.get_atoms()]) for res in pocket_residues], axis=0)
        full_coords = np.concatenate([np.array([atom.coord for atom in res.get_atoms()]) for res in pocket_residues], axis=0)
        full_atoms_names = np.concatenate([np.array([atom.get_id() for atom in res.get_atoms()]) for res in pocket_residues], axis=0)
        pocket_AA = np.concatenate([([three_to_one(atom.get_parent().get_resname()) for atom in res.get_atoms()]) for res in pocket_residues], axis=0)
        
        # removing Hs if present
        if remove_H:
            h_mask = full_atoms == 'H'
            full_atoms = full_atoms[~h_mask]
            pocket_coords = full_coords[~h_mask]
            full_atoms_names = full_atoms_names[~h_mask]
            pocket_AA = pocket_AA[~h_mask]
        
        try:
            pocket_one_hot = []
            for i in range(len(full_atoms)):
                a = full_atoms[i]
                aa = pocket_AA[i]
                atom_onehot = np.eye(1, len(pocket_atom_dict), pocket_atom_dict[a.capitalize()]).squeeze()
                amino_onehot = np.eye(1, len(amino_acid_dict), amino_acid_dict[aa.capitalize()]).squeeze()
                is_backbone = 1 if full_atoms_names[i].capitalize() in ['N','CA','C','O'] else 0
                pocket_one_hot.append(np.concatenate([atom_onehot, amino_onehot, (is_backbone,)]))
                
            pocket_one_hot = np.stack(pocket_one_hot)
        except KeyError as e:
            raise KeyError(
            f'{e} not in atom dict ({pdbfile})')
        
    pocket_one_hot = np.array(pocket_one_hot)
    return pocket_one_hot, pocket_coords, lig_coords